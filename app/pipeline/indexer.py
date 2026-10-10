"""Embeds each raw video's clips (cut on the fly, never stored) and upserts them into Qdrant.

Each point carries up to three independent named vectors: "visual" (CLIP,
always present), "transcript" (sentence-embedded Whisper transcript,
present only on clips where VAD-gated transcription found real speech),
and "ocr_text" (sentence-embedded on-screen text, present only on clips
where OCR found real text -- see vault note "OCR effort kicked off" entry,
2026-09-13). See vault note "Feasibility study" entry (2026-09-03) for why
transcript/ocr_text live in separate vector spaces rather than one blended
embedding.

Payload carries video_key (the raw video in S3)/start_ts/end_ts (the clip's window in it)/source_id so search can combine
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
from pathlib import PurePosixPath

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import (
    CLIP_DURATION_SECONDS,
    CLIP_EMBED_DIM,
    ENABLE_AUDIO_SEARCH,
    ENABLE_OCR_SEARCH,
    FRAMES_PER_CLIP,
    OCR_FRAMES_PER_CLIP,
    PAYLOAD_INDEX_FIELDS,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    RAW_VIDEOS_BUCKET,
    TRANSCRIPT_EMBED_DIM,
)
from app.pipeline.chunker import START_TS_DIGITS, cut_video
from app.pipeline.embedder import embed_clip, sample_frames

# Clips are named <video key minus extension>_clip<start_ts>.mp4 by app.pipeline.chunker;
# the regex pulls the start second back out of that name.
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
        # index_video() call, i.e. every ingest) would re-run a full
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


def _video_stem(video_key: str) -> str:
    """`cam1/clip.mp4` -> `cam1/clip`: the default source_id, and the base of each clip's
    virtual name (`cam1/clip_clip000010.mp4`). Includes the prefix so two cameras that both
    export a file called clip.mp4 stay distinct."""
    return PurePosixPath(video_key).with_suffix("").as_posix()


def _point_id(video_key: str, start_ts: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"s3://{RAW_VIDEOS_BUCKET}/{video_key}#t={start_ts}"))


def _video_filter(video_key: str) -> Filter:
    return Filter(must=[FieldCondition(key="video_key", match=MatchValue(value=video_key))])


def _existing_metadata(client: QdrantClient, video_key: str) -> dict:
    """source_id/tags/attributes of a video's current points, so re-indexing it (e.g. via
    /index, which carries no metadata) doesn't silently reset what an earlier /ingest set."""
    if not client.collection_exists(QDRANT_COLLECTION):
        return {}
    points, _ = client.scroll(QDRANT_COLLECTION, scroll_filter=_video_filter(video_key), limit=1, with_payload=True)
    if not points:
        return {}
    payload = points[0].payload
    return {k: payload[k] for k in ("source_id", "tags", "attributes") if k in payload}


def index_video(
    video_key: str,
    source_id: str | None = None,
    tags: list[str] | None = None,
    attributes: dict | None = None,
) -> int:
    """Embed one raw video's clips and replace that video's points in Qdrant.

    Clips are cut into a temp directory, embedded, and thrown away -- nothing is stored in
    S3. Each point records the raw `video_key` and the clip's `start_ts`/`end_ts`; the clip
    itself is cut again on demand when played (app.pipeline.clip_service).

    All clips are embedded *before* the video's old points are deleted and the new ones
    upserted, so a failure part-way leaves the previous index for that video untouched, and
    re-ingesting a changed video can't leave stale points behind (e.g. clips of an older,
    longer version).

    `source_id`/`tags`/`attributes` left as None are inherited from the video's existing
    points when it has any (falling back to a filename-derived source_id, no tags, no
    attributes); a value passed explicitly always wins. `clip_path` is a virtual name
    (`<stem>_clip<start_ts>.mp4`) kept for compatibility -- no object by that name exists.
    """
    client = get_client()
    ensure_collection(client)
    stem = _video_stem(video_key)

    inherited = _existing_metadata(client, video_key)
    # Truthy check for source_id, not `is not None`: an empty string means "not given",
    # matching search.py's `_filter_conditions()` check on the other side.
    source_id = source_id or inherited.get("source_id") or stem
    tags = tags if tags is not None else inherited.get("tags", [])
    attributes = attributes if attributes is not None else inherited.get("attributes", {})

    points = []
    with cut_video(video_key) as clip_paths:
        for clip_path in clip_paths:
            match = _CLIP_NAME_RE.match(clip_path.stem)
            start_ts = int(match.group("start_ts")) if match else 0
            print(f"Embedding {stem} @ {start_ts}s ...")
            # Sampled once here (not left to embed_clip's own internal default) so OCR
            # below can reuse these already-decoded frames via a subsample instead of
            # opening and seeking the video file a second time.
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

                # Subsampled from the already-decoded `frames` above: OCR wants a sparser
                # sample than the visual embedding (on-screen text is usually static across
                # consecutive frames), but there's no need to re-read the file for fewer.
                step = max(len(frames) // OCR_FRAMES_PER_CLIP, 1)
                ocr_frames = frames[::step][:OCR_FRAMES_PER_CLIP]
                ocr_text = extract_text(ocr_frames) or None
                if ocr_text:
                    print(f"  -> on-screen text detected: {ocr_text!r}")
                    vectors["ocr_text"] = embed_general_text(ocr_text).tolist()

            points.append(
                PointStruct(
                    id=_point_id(video_key, start_ts),
                    vector=vectors,
                    payload={
                        "video_key": video_key,
                        "clip_path": f"{stem}_clip{start_ts:0{START_TS_DIGITS}d}.mp4",
                        "source_id": source_id,
                        "start_ts": start_ts,
                        "end_ts": start_ts + CLIP_DURATION_SECONDS,
                        "has_speech": transcript is not None,
                        "transcript": transcript or "",
                        "has_text": ocr_text is not None,
                        "ocr_text": ocr_text or "",
                        "tags": tags,
                        "attributes": attributes,
                        # Used by app.pipeline.sources (visibility) and app.pipeline.retention
                        # (expiry) -- ISO 8601 UTC, re-stamped on every re-index.
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

    client.delete(collection_name=QDRANT_COLLECTION, points_selector=FilterSelector(filter=_video_filter(video_key)))
    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"Indexed {len(points)} clips of {video_key} into '{QDRANT_COLLECTION}'.")
    return len(points)
