"""Core retrieval: text query -> CLIP text embedding -> Qdrant similarity search.

No LLM in this path -- purely nearest-neighbor over embeddings, per the
project's core design constraint. Optionally fuses in one or more
secondary signals -- transcribed speech content (see vault note
"Feasibility study" entry, 2026-09-03) and/or on-screen OCR'd text (see
vault note "OCR effort kicked off" entry, 2026-09-13) -- via Qdrant's
native RRF fusion over named vectors on the same point, N-way, no
hand-rolled fusion logic needed.
"""

from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchAny, MatchValue, Prefetch

from app import storage
from app.config import (
    ATTRIBUTE_VERIFICATION_FRAMES,
    ATTRIBUTE_VERIFICATION_POOL,
    ENABLE_ATTRIBUTE_VERIFICATION,
    MIN_OCR_SCORE,
    MIN_SIMILARITY_SCORE,
    MIN_TRANSCRIPT_SCORE,
    QDRANT_COLLECTION,
)
from app.pipeline.embedder import embed_text, sample_frames
from app.pipeline.indexer import get_client
from app.pipeline.query_parser import extract_attribute_object_pairs


_SUPPORTED_FILTER_OPS = {"eq", "in"}


def _filter_conditions(source_id: str | None, filters: list[dict] | None) -> list[FieldCondition]:
    """Builds the list of Qdrant FieldConditions from the generic `filters`
    list (see vault note "Plug-and-play audit" entry, 2026-09-13), plus the
    `source_id` convenience param folded in as an "eq" condition on it.

    `filters` entries are `{"field": <payload key, dotted for nested
    attributes -- e.g. "attributes.location">, "op": "eq" | "in", "value":
    ...}`. Just these two ops for now (exact match, match-any) -- covers
    every use case actually needed by tags/attributes/source_id filtering
    today; range ops (for numeric/date attributes) are a real gap but
    nothing in this project's current deployments needs them yet, so
    they're left for whenever a real one does rather than built speculatively.
    """
    conditions = []
    if source_id:
        conditions.append(FieldCondition(key="source_id", match=MatchValue(value=source_id)))
    for f in filters or []:
        field, op, value = f["field"], f["op"], f["value"]
        if op == "eq":
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        elif op == "in":
            conditions.append(FieldCondition(key=field, match=MatchAny(any=value)))
        else:
            raise ValueError(f"Unsupported filter op {op!r} (supported: {_SUPPORTED_FILTER_OPS})")
    return conditions


def _to_result(hit) -> dict:
    key = hit.payload.get("clip_path")
    return {
        "score": hit.score,
        "clip_path": key,
        "clip_url": storage.presigned_clip_url(key) if key else None,
        "source_id": hit.payload.get("source_id"),
        "start_ts": hit.payload.get("start_ts"),
        "end_ts": hit.payload.get("end_ts"),
        "has_speech": hit.payload.get("has_speech", False),
        "transcript": hit.payload.get("transcript", ""),
        "has_text": hit.payload.get("has_text", False),
        "ocr_text": hit.payload.get("ocr_text", ""),
        "tags": hit.payload.get("tags", []),
        "attributes": hit.payload.get("attributes", {}),
    }


def _rerank_by_attribute(results: list[dict], attribute: str, obj: str, top_k: int) -> list[dict]:
    """Re-ranks `results` by verifying (attribute, obj) against each clip's
    sampled frames via app.pipeline.object_verifier (see vault note
    "Attribute-binding / bag-of-words retrieval failure", 2026-09-08).
    Requires `results` to already have `ATTRIBUTE_VERIFICATION_POOL`-many
    candidates for this to have room to find a better ordering than the
    plain visual ranking alone.
    """
    from app.pipeline.object_verifier import verify_attribute_in_frames

    for result in results:
        with storage.local_clip(result["clip_path"]) as local_path:
            frames = sample_frames(local_path, n_frames=ATTRIBUTE_VERIFICATION_FRAMES)
        result["attribute_score"] = verify_attribute_in_frames(frames, attribute, obj)

    results.sort(key=lambda r: r["attribute_score"], reverse=True)
    return results[:top_k]


def search_clips(
    query: str,
    top_k: int = 5,
    source_id: str | None = None,
    filters: list[dict] | None = None,
    use_transcript_fusion: bool = False,
    use_ocr_fusion: bool = False,
    verify_attributes: bool | None = None,
) -> list[dict]:
    if verify_attributes is None:
        verify_attributes = ENABLE_ATTRIBUTE_VERIFICATION
    attribute_pairs = extract_attribute_object_pairs(query) if verify_attributes else []
    # Only the first (attribute, object) pair is handled -- a query naming
    # more than one attribute-object pair (rare for this project's short,
    # CCTV-style queries) falls back to using just the first one found.
    fetch_limit = max(top_k, ATTRIBUTE_VERIFICATION_POOL) if attribute_pairs else top_k

    client = get_client()
    # Nothing indexed yet (the collection is only created by the indexer) ->
    # no matches, rather than a Qdrant 404 surfacing as a 500.
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    # Computed once and reused below (both for query_filter here and for
    # the fusion branch's speech_filter_conditions) rather than calling
    # _filter_conditions() a second time with the same arguments.
    base_conditions = _filter_conditions(source_id, filters)
    query_filter = Filter(must=base_conditions) if base_conditions else None
    visual_vector = embed_text(query)

    if not use_transcript_fusion and not use_ocr_fusion:
        # score_threshold drops hits below MIN_SIMILARITY_SCORE -- without
        # this, Qdrant always returns the top_k nearest points regardless of
        # how poor a match they are, so an out-of-distribution query
        # (something not actually in the footage) would otherwise return a
        # confident-looking false positive instead of "no match". See
        # app.config.MIN_SIMILARITY_SCORE for how the cutoff was derived.
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=visual_vector.tolist(),
            using="visual",
            query_filter=query_filter,
            limit=fetch_limit,
            score_threshold=MIN_SIMILARITY_SCORE,
        ).points
        results = [_to_result(h) for h in hits]
    else:
        # N-way fusion: combine the visual ranking with a ranking over
        # each requested secondary signal (transcript and/or ocr_text),
        # via Reciprocal Rank Fusion. Every branch is gated by its own
        # calibrated threshold (MIN_SIMILARITY_SCORE / MIN_TRANSCRIPT_SCORE
        # / MIN_OCR_SCORE) -- ungating a branch (the original transcript
        # design) measurably hurt precision by letting weak/off-topic
        # matches into the fused ranking (see eval results, 2026-09-09).
        candidate_pool = max(fetch_limit * 4, 20)
        prefetches = [
            Prefetch(
                query=visual_vector.tolist(),
                using="visual",
                filter=query_filter,
                score_threshold=MIN_SIMILARITY_SCORE,
                limit=candidate_pool,
            ),
        ]

        if use_transcript_fusion:
            from app.pipeline.text_embedder import embed_transcript_text

            speech_filter_conditions = [
                FieldCondition(key="has_speech", match=MatchValue(value=True)),
                *base_conditions,
            ]
            prefetches.append(
                Prefetch(
                    query=embed_transcript_text(query).tolist(),
                    using="transcript",
                    filter=Filter(must=speech_filter_conditions),
                    score_threshold=MIN_TRANSCRIPT_SCORE,
                    limit=candidate_pool,
                )
            )

        if use_ocr_fusion:
            from app.pipeline.text_embedder import embed_text as embed_general_text

            text_filter_conditions = [
                FieldCondition(key="has_text", match=MatchValue(value=True)),
                *base_conditions,
            ]
            prefetches.append(
                Prefetch(
                    query=embed_general_text(query).tolist(),
                    using="ocr_text",
                    filter=Filter(must=text_filter_conditions),
                    score_threshold=MIN_OCR_SCORE,
                    limit=candidate_pool,
                )
            )

        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
        ).points
        results = [_to_result(h) for h in hits]

    if attribute_pairs:
        attribute, obj = attribute_pairs[0]
        results = _rerank_by_attribute(results, attribute, obj, top_k)
    else:
        results = results[:top_k]

    return results
