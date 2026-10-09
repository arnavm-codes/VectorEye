"""Embeds all chunked clips and upserts them into Qdrant.

Each point carries up to three independent named vectors: "visual" (CLIP,
always present), "transcript" (sentence-embedded Whisper transcript,
present only on clips where VAD-gated transcription found real speech),
and "ocr_text" (sentence-embedded on-screen text, present only on clips
where OCR found real text -- see vault note "OCR effort kicked off" entry,
2026-09-13). See vault note "Feasibility study" entry (2026-09-03) for why
transcript/ocr_text live in separate vector spaces rather than one blended
embedding.

Payload carries clip_path (the clip's S3 object key in the clips bucket)/source_id/start_ts/end_ts so search can combine
vector similarity with structured filtering (see vault note "Vector DB"
section for why Qdrant was fixated for this), plus has_speech/transcript
for the audio signal, plus generic tags/attributes (see vault note
"Plug-and-play audit" entry, 2026-09-13) for arbitrary deployment-specific
metadata (trip name, location, course id, ...) beyond the one required
source_id grouping key.
"""

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from app import storage
from app.config import (
    CLIP_DURATION_SECONDS,
    CLIP_EMBED_DIM,
    CLIPS_BUCKET,
    ENABLE_AUDIO_SEARCH,
    ENABLE_OCR_SEARCH,
    FRAMES_PER_CLIP,
    OCR_FRAMES_PER_CLIP,
    PAYLOAD_INDEX_FIELDS,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    TRANSCRIPT_EMBED_DIM,
)
from app.pipeline.embedder import embed_clip, sample_frames

# `source_id` is the fallback-only naming convention: <source_id>_clip<start_ts>.mp4,
# produced by app.pipeline.chunker for videos that went through the CLI/legacy
# path with no explicit metadata supplied. Callers that pass `source_id`
# explicitly to index_clips() (the ingestion-API path) skip this entirely.
_CLIP_NAME_RE = re.compile(r"^(?P<source_id>.+)_clip(?P<start_ts>\d+)$")


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _migrate_legacy_camera_id(client: QdrantClient):
    """One-time migration for collections indexed before the camera_id ->
    source_id payload rename (see vault note "Plug-and-play audit" entry,
    2026-09-13, and code-review finding the same day). Without this, a
    pre-existing point that only has the old `camera_id` key is silently
    invisible to every source_id filter and to the Streamlit source
    dropdown -- no error, just missing results -- until manually re-indexed.
    Copies `camera_id`'s value into `source_id` wherever `source_id` is
    still missing; idempotent, since a fully-migrated collection has
    nothing left matching the scan and this becomes a fast no-op.

    Client-side scan + per-point set_payload, not a server-side bulk
    update -- simplest correct approach at this project's scale (POC-sized
    collections, a few thousand points at most); would want a proper
    batched/filtered update if this ever ran against a much larger
    collection.
    """
    migrated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            if "source_id" not in p.payload and "camera_id" in p.payload:
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload={"source_id": p.payload["camera_id"]},
                    points=[p.id],
                )
                migrated += 1
        if offset is None:
            break
    if migrated:
        print(f"Migrated {migrated} legacy camera_id-only point(s) to source_id in '{QDRANT_COLLECTION}'.")


_migration_checked: set[str] = set()


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "visual": VectorParams(size=CLIP_EMBED_DIM, distance=Distance.COSINE),
                "transcript": VectorParams(size=TRANSCRIPT_EMBED_DIM, distance=Distance.COSINE),
                # Same embedding model/dim as "transcript" (see
                # app.pipeline.text_embedder) -- a separate named vector
                # because it's a different signal (on-screen vs. spoken
                # text), not because it needs a different embedding space.
                "ocr_text": VectorParams(size=TRANSCRIPT_EMBED_DIM, distance=Distance.COSINE),
            },
        )
    elif QDRANT_COLLECTION not in _migration_checked:
        # Without this guard, ensure_collection() (called on every single
        # index_clips() call, i.e. every ingest) would re-run a full
        # paginated scroll of the ENTIRE collection every time -- fine at
        # this project's ~10-point test scale, but directly defeats Phase
        # 4's whole purpose (cheap repeated watch-folder polling against a
        # collection meant to keep growing). Found via integration testing
        # after Phase 4, not caught by any single phase's isolated tests
        # since none of them grew the collection large enough to notice.
        # Once-per-process-per-collection is enough: a collection that
        # still has legacy camera_id-only points after its first migration
        # pass in this process's lifetime would need a fresh process
        # restart to pick up any further (externally-added) legacy points
        # -- an acceptable gap for a migration meant to run once, ever.
        _migrate_legacy_camera_id(client)

        # Same reasoning as the camera_id migration above, for a different
        # breaking schema change: create_collection() only runs for a
        # brand-new collection, so an *existing* collection indexed before
        # ENABLE_OCR_SEARCH was turned on never gets the "ocr_text" vector
        # added -- Qdrant has no API to add a named vector to an existing
        # collection. Without this check, that surfaces as an opaque
        # "unknown vector name" error deep inside client.upsert() (or, at
        # query time, inside app.api.search's ocr_text Prefetch) instead of
        # a clear message pointing at the actual fix. Found by code review,
        # not by this session's own testing, which always tested OCR
        # against a freshly dropped-and-recreated collection (the same
        # `git log`-documented precedent this error message points to).
        if ENABLE_OCR_SEARCH:
            existing_vectors = client.get_collection(QDRANT_COLLECTION).config.params.vectors
            if "ocr_text" not in existing_vectors:
                raise RuntimeError(
                    f"ENABLE_OCR_SEARCH is on, but collection '{QDRANT_COLLECTION}' was created "
                    "before OCR support and has no 'ocr_text' vector. Qdrant can't add a named "
                    "vector to an existing collection -- drop and re-create it (same as the "
                    "transcript-vector addition on 2026-09-03), then re-index: "
                    f"QdrantClient(...).delete_collection('{QDRANT_COLLECTION}')."
                )
        _migration_checked.add(QDRANT_COLLECTION)
    # Payload indexes are additive/idempotent (a re-run against an existing
    # collection just confirms the index already exists) -- calling this
    # every time, not just on first creation, means adding a field to
    # PAYLOAD_INDEX_FIELDS and re-running the pipeline is enough to pick it
    # up, no separate migration step. Every currently-indexed field is a
    # flat string/keyword (source_id, tags, or a dotted attributes.* path);
    # a deployment indexing a numeric/date attribute for range filtering
    # would need a different PayloadSchemaType, not handled generically yet.
    for field in PAYLOAD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def _parse_clip_metadata(clip_path: Path) -> dict:
    match = _CLIP_NAME_RE.match(clip_path.stem)
    if not match:
        return {"source_id": clip_path.stem, "start_ts": 0, "end_ts": CLIP_DURATION_SECONDS}
    # The number in the filename is the clip's actual start second in the
    # source video (see chunker.py) -- not a sequential index -- since two
    # overlapping chunking passes (offsets 0 and CHUNK_OVERLAP_SECONDS) share
    # this naming scheme and can't be told apart by a plain sequence number.
    start_ts = int(match.group("start_ts"))
    return {
        "source_id": match.group("source_id"),
        "start_ts": start_ts,
        "end_ts": start_ts + CLIP_DURATION_SECONDS,
    }


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"s3://{CLIPS_BUCKET}/{key}"))


def index_clips(
    only_new: bool = True,
    video_stem: str | None = None,
    keys: list[str] | None = None,
    source_id: str | None = None,
    tags: list[str] | None = None,
    attributes: dict | None = None,
) -> int:
    """Embed clips from the clips bucket and upsert into Qdrant.

    By default (`keys` omitted) every clip in the bucket is a candidate -- the
    bulk `/index` behavior. Passing an explicit `keys` list indexes just those
    clips without listing the bucket -- the path the ingestion API uses so
    ingesting one new video doesn't touch every other clip already indexed.

    only_new=True skips clips whose point already exists (cheap re-runs);
    pass False after changing the embedding backend or models, or when
    re-ingesting a changed video.
    video_stem limits the pass to clips cut from that source video (clip
    names are `<stem>_clip<start_ts>.mp4`) instead of the whole bucket.

    `source_id`/`tags`/`attributes` are optional explicit metadata applied to
    every clip indexed by this call. When `source_id` is omitted, each clip's
    source_id/start_ts/end_ts are derived from its filename.
    """
    client = get_client()
    ensure_collection(client)

    if keys is None:
        keys = storage.list_clips()
    if video_stem is not None:
        keys = [k for k in keys if _parse_clip_metadata(Path(k))["source_id"] == video_stem]
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
            # Sampled once here (not left to embed_clip's own internal
            # default) so OCR below can reuse these already-decoded frames
            # via a subsample instead of opening and seeking the video file
            # a second time. See embed_clip()'s docstring for why this matters.
            frames = sample_frames(clip_path, n_frames=FRAMES_PER_CLIP)
            visual_vector = embed_clip(clip_path, frames=frames)
            vectors = {"visual": visual_vector.tolist()}

            transcript = None
            if ENABLE_AUDIO_SEARCH:
                from app.pipeline.transcriber import transcribe_clip

                transcript = transcribe_clip(clip_path)
                if transcript:
                    from app.pipeline.text_embedder import embed_transcript_text

                    print(f"  -> speech detected: {transcript!r}")
                    vectors["transcript"] = embed_transcript_text(transcript).tolist()

            ocr_text = None
            if ENABLE_OCR_SEARCH:
                from app.pipeline.ocr import extract_text
                from app.pipeline.text_embedder import embed_text as embed_general_text

                # Subsampled from the already-decoded `frames` above, not a
                # fresh sample_frames() call -- OCR wants a sparser sample than
                # the visual embedding (on-screen text is typically static
                # across many consecutive frames), but there's no need to
                # re-open and re-seek the video file to get fewer frames from
                # it when the denser set already covers the same span.
                step = max(len(frames) // OCR_FRAMES_PER_CLIP, 1)
                ocr_frames = frames[::step][:OCR_FRAMES_PER_CLIP]
                ocr_text = extract_text(ocr_frames) or None
                if ocr_text:
                    print(f"  -> on-screen text detected: {ocr_text!r}")
                    vectors["ocr_text"] = embed_general_text(ocr_text).tolist()

        meta = _parse_clip_metadata(Path(key))
        # Truthy check, not `is not None` -- an empty string must be treated
        # the same as omitted (fall back to the filename-derived source_id),
        # matching search.py's `_filter_conditions()` check on the other side.
        if source_id:
            meta["source_id"] = source_id
        points.append(
            PointStruct(
                id=_point_id(key),
                vector=vectors,
                payload={
                    # S3 key in the clips bucket (field name kept for compatibility).
                    "clip_path": key,
                    "has_speech": transcript is not None,
                    "transcript": transcript or "",
                    "has_text": ocr_text is not None,
                    "ocr_text": ocr_text or "",
                    "tags": tags or [],
                    "attributes": attributes or {},
                    # Used by app.pipeline.sources (Phase 5 visibility) and
                    # app.pipeline.retention (Phase 5 retention sweep) to
                    # report/age clips -- ISO 8601 UTC, re-stamped on every
                    # re-index (a re-ingested clip's age resets, consistent
                    # with it being freshly (re-)processed).
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
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
