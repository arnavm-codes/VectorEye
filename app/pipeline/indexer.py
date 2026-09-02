"""Embeds all chunked clips and upserts them into Qdrant.

Payload carries clip_path/camera_id/start_ts/end_ts so search can combine
vector similarity with structured filtering (see vault note "Vector DB"
section for why Qdrant was fixated for this).
"""

import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import (
    CLIP_DURATION_SECONDS,
    CLIP_EMBED_DIM,
    CLIPS_DIR,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from app.pipeline.embedder import embed_clip

_CLIP_NAME_RE = re.compile(r"^(?P<camera_id>.+)_clip(?P<index>\d+)$")


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=CLIP_EMBED_DIM, distance=Distance.COSINE),
        )


def _parse_clip_metadata(clip_path: Path) -> dict:
    match = _CLIP_NAME_RE.match(clip_path.stem)
    if not match:
        return {"camera_id": clip_path.stem, "index": 0}
    index = int(match.group("index"))
    return {
        "camera_id": match.group("camera_id"),
        "index": index,
        "start_ts": index * CLIP_DURATION_SECONDS,
        "end_ts": (index + 1) * CLIP_DURATION_SECONDS,
    }


def index_clips(clips_dir: Path = CLIPS_DIR) -> int:
    client = get_client()
    ensure_collection(client)

    clip_paths = sorted(clips_dir.glob("*.mp4"))
    points = []
    for clip_path in clip_paths:
        print(f"Embedding {clip_path.name} ...")
        vector = embed_clip(clip_path)
        meta = _parse_clip_metadata(clip_path)
        # Deterministic ID from clip_path (not a fresh uuid4 every run) so
        # re-running the pipeline upserts/overwrites existing clips instead
        # of duplicating them in the collection.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(clip_path)))
        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"clip_path": str(clip_path), **meta},
            )
        )

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"Indexed {len(points)} clips into '{QDRANT_COLLECTION}'.")
    return len(points)


if __name__ == "__main__":
    index_clips()
