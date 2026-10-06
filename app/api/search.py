"""Core retrieval: text query -> CLIP text embedding -> Qdrant similarity search.

No LLM in this path -- purely nearest-neighbor over embeddings, per the
project's core design constraint. Optionally fuses in a second signal from
transcribed speech content (see vault note "Feasibility study" entry,
2026-09-03) via Qdrant's native RRF fusion over two named vectors on the
same point -- no hand-rolled fusion logic needed.
"""

from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchValue, Prefetch

from app import storage
from app.config import (
    ATTRIBUTE_VERIFICATION_FRAMES,
    ATTRIBUTE_VERIFICATION_POOL,
    ENABLE_ATTRIBUTE_VERIFICATION,
    MIN_SIMILARITY_SCORE,
    MIN_TRANSCRIPT_SCORE,
    QDRANT_COLLECTION,
)
from app.pipeline.embedder import embed_text, sample_frames
from app.pipeline.indexer import get_client
from app.pipeline.query_parser import extract_attribute_object_pairs


def _camera_filter(camera_id: str | None) -> Filter | None:
    if not camera_id:
        return None
    return Filter(must=[FieldCondition(key="camera_id", match=MatchValue(value=camera_id))])


def _to_result(hit) -> dict:
    key = hit.payload.get("clip_path")
    return {
        "score": hit.score,
        "clip_path": key,
        "clip_url": storage.presigned_clip_url(key) if key else None,
        "camera_id": hit.payload.get("camera_id"),
        "start_ts": hit.payload.get("start_ts"),
        "end_ts": hit.payload.get("end_ts"),
        "has_speech": hit.payload.get("has_speech", False),
        "transcript": hit.payload.get("transcript", ""),
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
    camera_id: str | None = None,
    use_transcript_fusion: bool = False,
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
    query_filter = _camera_filter(camera_id)
    visual_vector = embed_text(query)

    if not use_transcript_fusion:
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
        # Transcript fusion: combine the visual ranking with a ranking over
        # the (much smaller) set of clips that have a transcript, via
        # Reciprocal Rank Fusion. Both branches are gated by their own
        # calibrated threshold (MIN_SIMILARITY_SCORE / MIN_TRANSCRIPT_SCORE)
        # -- ungating the transcript branch (the original design) measurably
        # hurt precision by letting weak/off-topic transcript matches into
        # the fused ranking (see eval results, 2026-09-09).
        from app.pipeline.text_embedder import embed_transcript_text

        transcript_vector = embed_transcript_text(query)
        candidate_pool = max(fetch_limit * 4, 20)

        speech_filter_conditions = [
            FieldCondition(key="has_speech", match=MatchValue(value=True))
        ]
        if camera_id:
            speech_filter_conditions.append(
                FieldCondition(key="camera_id", match=MatchValue(value=camera_id))
            )

        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                Prefetch(
                    query=visual_vector.tolist(),
                    using="visual",
                    filter=query_filter,
                    score_threshold=MIN_SIMILARITY_SCORE,
                    limit=candidate_pool,
                ),
                Prefetch(
                    query=transcript_vector.tolist(),
                    using="transcript",
                    filter=Filter(must=speech_filter_conditions),
                    score_threshold=MIN_TRANSCRIPT_SCORE,
                    limit=candidate_pool,
                ),
            ],
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
