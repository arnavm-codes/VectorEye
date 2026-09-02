"""Core retrieval: text query -> CLIP text embedding -> Qdrant similarity search.

No LLM in this path -- purely nearest-neighbor over embeddings, per the
project's core design constraint.
"""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import QDRANT_COLLECTION
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

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=top_k,
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
