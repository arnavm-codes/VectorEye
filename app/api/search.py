"""Core retrieval: text query -> CLIP text embedding -> Qdrant similarity search.

No LLM in this path -- purely nearest-neighbor over embeddings, per the
project's core design constraint. Optionally fuses in a second signal from
transcribed speech content (see vault note "Feasibility study" entry,
2026-09-03) via Qdrant's native RRF fusion over two named vectors on the
same point -- no hand-rolled fusion logic needed.
"""

from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchValue, Prefetch

from app.config import MIN_SIMILARITY_SCORE, QDRANT_COLLECTION
from app.pipeline.embedder import embed_text
from app.pipeline.indexer import get_client


def _camera_filter(camera_id: str | None) -> Filter | None:
    if not camera_id:
        return None
    return Filter(must=[FieldCondition(key="camera_id", match=MatchValue(value=camera_id))])


def _to_result(hit) -> dict:
    return {
        "score": hit.score,
        "clip_path": hit.payload.get("clip_path"),
        "camera_id": hit.payload.get("camera_id"),
        "start_ts": hit.payload.get("start_ts"),
        "end_ts": hit.payload.get("end_ts"),
        "has_speech": hit.payload.get("has_speech", False),
        "transcript": hit.payload.get("transcript", ""),
    }


def search_clips(
    query: str,
    top_k: int = 5,
    camera_id: str | None = None,
    use_transcript_fusion: bool = False,
) -> list[dict]:
    client = get_client()
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
            limit=top_k,
            score_threshold=MIN_SIMILARITY_SCORE,
        ).points
        return [_to_result(h) for h in hits]

    # Transcript fusion: combine the visual ranking with a ranking over the
    # (much smaller) set of clips that have a transcript, via Reciprocal
    # Rank Fusion. Only the visual branch gets MIN_SIMILARITY_SCORE applied
    # -- the transcript-embedding score scale has never been calibrated
    # (see app.config.TRANSCRIPT_EMBED_MODEL), so gating it on an unproven
    # threshold would risk silently dropping the one signal this feature
    # exists to add. RRF only uses rank, not raw score, so an uncalibrated
    # scale doesn't distort the fused result the way a bad threshold would.
    from app.pipeline.text_embedder import embed_transcript_text

    transcript_vector = embed_transcript_text(query)
    candidate_pool = max(top_k * 4, 20)

    speech_filter_conditions = [FieldCondition(key="has_speech", match=MatchValue(value=True))]
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
                limit=candidate_pool,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points
    return [_to_result(h) for h in hits]
