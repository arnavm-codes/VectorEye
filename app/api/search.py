"""Core retrieval: text query -> CLIP text embedding -> Qdrant similarity search.

No LLM in this path -- purely nearest-neighbor over embeddings, per the
project's core design constraint.
"""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import MIN_SIMILARITY_SCORE, QDRANT_COLLECTION
from app.pipeline.embedder import embed_text
from app.pipeline.indexer import get_client


def search_clips(query: str, top_k: int = 5, camera_id: str | None = None) -> list[dict]:
    client = get_client()
    query_vector = embed_text(query)

    query_filter = None
    if camera_id:
        query_filter = Filter(
            must=[FieldCondition(key="camera_id", match=MatchValue(value=camera_id))]
        )

    # score_threshold drops hits below MIN_SIMILARITY_SCORE -- without this,
    # Qdrant always returns the top_k nearest points regardless of how poor
    # a match they are, so an out-of-distribution query (something not
    # actually in the footage) would otherwise return a confident-looking
    # false positive instead of "no match". See app.config.MIN_SIMILARITY_SCORE
    # for how the cutoff was derived.
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=top_k,
        score_threshold=MIN_SIMILARITY_SCORE,
    ).points

    return [
        {
            "score": hit.score,
            "clip_path": hit.payload.get("clip_path"),
            "camera_id": hit.payload.get("camera_id"),
            "start_ts": hit.payload.get("start_ts"),
            "end_ts": hit.payload.get("end_ts"),
        }
        for hit in hits
    ]
