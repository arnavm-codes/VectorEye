"""Embeds all chunked clips and upserts them into Qdrant.

Each point carries two independent named vectors: "visual" (CLIP, always
present) and "transcript" (sentence-embedded Whisper transcript, present
only on clips where VAD-gated transcription found real speech). See vault
note "Feasibility study" entry (2026-09-03) for why these live in separate
vector spaces rather than one blended embedding.

Payload carries clip_path/camera_id/start_ts/end_ts so search can combine
vector similarity with structured filtering (see vault note "Vector DB"
section for why Qdrant was fixated for this), plus has_speech/transcript
for the audio signal.
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
    ENABLE_AUDIO_SEARCH,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    TRANSCRIPT_EMBED_DIM,
)
from app.pipeline.embedder import embed_clip

_CLIP_NAME_RE = re.compile(r"^(?P<camera_id>.+)_clip(?P<start_ts>\d+)$")


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "visual": VectorParams(size=CLIP_EMBED_DIM, distance=Distance.COSINE),
                "transcript": VectorParams(size=TRANSCRIPT_EMBED_DIM, distance=Distance.COSINE),
            },
        )


def _parse_clip_metadata(clip_path: Path) -> dict:
    match = _CLIP_NAME_RE.match(clip_path.stem)
    if not match:
        return {"camera_id": clip_path.stem, "start_ts": 0, "end_ts": CLIP_DURATION_SECONDS}
    # The number in the filename is the clip's actual start second in the
    # source video (see chunker.py) -- not a sequential index -- since two
    # overlapping chunking passes (offsets 0 and CHUNK_OVERLAP_SECONDS) share
    # this naming scheme and can't be told apart by a plain sequence number.
    start_ts = int(match.group("start_ts"))
    return {
        "camera_id": match.group("camera_id"),
        "start_ts": start_ts,
        "end_ts": start_ts + CLIP_DURATION_SECONDS,
    }


def index_clips(clips_dir: Path = CLIPS_DIR) -> int:
    client = get_client()
    ensure_collection(client)

    clip_paths = sorted(clips_dir.glob("*.mp4"))
    points = []
    for clip_path in clip_paths:
        print(f"Embedding {clip_path.name} ...")
        visual_vector = embed_clip(clip_path)
        vectors = {"visual": visual_vector.tolist()}

        transcript = None
        if ENABLE_AUDIO_SEARCH:
            from app.pipeline.transcriber import transcribe_clip

            transcript = transcribe_clip(clip_path)
            if transcript:
                from app.pipeline.text_embedder import embed_transcript_text

                print(f"  -> speech detected: {transcript!r}")
                vectors["transcript"] = embed_transcript_text(transcript).tolist()

        meta = _parse_clip_metadata(clip_path)
        # Deterministic ID from clip_path (not a fresh uuid4 every run) so
        # re-running the pipeline upserts/overwrites existing clips instead
        # of duplicating them in the collection.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(clip_path)))
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "clip_path": str(clip_path),
                    "has_speech": transcript is not None,
                    "transcript": transcript or "",
                    **meta,
                },
            )
        )

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"Indexed {len(points)} clips into '{QDRANT_COLLECTION}'.")
    return len(points)


if __name__ == "__main__":
    index_clips()
