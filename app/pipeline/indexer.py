"""Embeds all chunked clips and upserts them into Qdrant.

Each point carries two independent named vectors: "visual" (CLIP, always
present) and "transcript" (sentence-embedded Whisper transcript, present
only on clips where VAD-gated transcription found real speech). See vault
note "Feasibility study" entry (2026-09-03) for why these live in separate
vector spaces rather than one blended embedding.

Payload carries clip_path (the clip's S3 object key in the clips bucket)/camera_id/start_ts/end_ts so search can combine
vector similarity with structured filtering (see vault note "Vector DB"
section for why Qdrant was fixated for this), plus has_speech/transcript
for the audio signal.
"""

import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import storage
from app.config import (
    CLIP_DURATION_SECONDS,
    CLIP_EMBED_DIM,
    CLIPS_BUCKET,
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


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"s3://{CLIPS_BUCKET}/{key}"))


def index_clips(only_new: bool = True, video_stem: str | None = None) -> int:
    """Embed every clip in the clips bucket and upsert into Qdrant.
    only_new=True skips clips whose point already exists (cheap re-runs);
    pass False after changing the embedding backend or models.
    video_stem limits the pass to clips cut from that source video (clip
    names are `<stem>_clip<start_ts>.mp4`) instead of the whole bucket."""
    client = get_client()
    ensure_collection(client)

    keys = storage.list_clips()
    if video_stem is not None:
        keys = [k for k in keys if _parse_clip_metadata(Path(k))["camera_id"] == video_stem]
    if only_new and keys:
        existing: set[str] = set()
        for i in range(0, len(keys), 256):  # batch to keep requests small on large libraries
            existing |= {
                str(p.id)
                for p in client.retrieve(
                    collection_name=QDRANT_COLLECTION,
                    ids=[_point_id(k) for k in keys[i : i + 256]],
                    with_payload=False,
                    with_vectors=False,
                )
            }
        keys = [k for k in keys if _point_id(k) not in existing]

    points = []
    for key in keys:
        print(f"Embedding {key} ...")
        with storage.local_clip(key) as clip_path:
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

        meta = _parse_clip_metadata(Path(key))
        points.append(
            PointStruct(
                id=_point_id(key),
                vector=vectors,
                payload={
                    # S3 key in the clips bucket (field name kept for compatibility).
                    "clip_path": key,
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
