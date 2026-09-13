"""Embeds all chunked clips and upserts them into Qdrant.

Each point carries two independent named vectors: "visual" (CLIP, always
present) and "transcript" (sentence-embedded Whisper transcript, present
only on clips where VAD-gated transcription found real speech). See vault
note "Feasibility study" entry (2026-09-03) for why these live in separate
vector spaces rather than one blended embedding.

Payload carries clip_path/source_id/start_ts/end_ts so search can combine
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

from app.config import (
    CLIP_DURATION_SECONDS,
    CLIP_EMBED_DIM,
    CLIPS_DIR,
    ENABLE_AUDIO_SEARCH,
    PAYLOAD_INDEX_FIELDS,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    TRANSCRIPT_EMBED_DIM,
)
from app.pipeline.embedder import embed_clip

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


def index_clips(
    clips_dir: Path = CLIPS_DIR,
    clip_paths: list[Path] | None = None,
    source_id: str | None = None,
    tags: list[str] | None = None,
    attributes: dict | None = None,
) -> int:
    """Embeds and indexes clips.

    By default (`clip_paths` omitted), embeds and indexes every `*.mp4` in
    `clips_dir` -- the CLI/`scripts/run_pipeline.py` behavior. Passing an
    explicit `clip_paths` list instead indexes just those clips without
    scanning/re-globbing the directory -- the path the ingestion API (see
    vault note "Plug-and-play audit" entry, 2026-09-13) uses so ingesting
    one new video doesn't re-embed every other clip already in `clips_dir`.

    `source_id`/`tags`/`attributes` are optional explicit metadata applied
    to every clip indexed by this call -- the generic path a caller (e.g.
    the ingestion API) uses instead of relying on the legacy
    filename-parsing fallback in `_parse_clip_metadata`. When `source_id`
    is omitted, each clip's source_id/start_ts/end_ts are still derived
    from its filename as before, for backward compatibility with the
    CLI/`scripts/run_pipeline.py` path.
    """
    client = get_client()
    ensure_collection(client)

    if clip_paths is None:
        clip_paths = sorted(clips_dir.glob("*.mp4"))
    points = []
    for clip_path in clip_paths:
        # Resolve to an absolute path before it's used for anything
        # identity-related (the point ID hash and the stored clip_path) --
        # a relative `clips_dir` (e.g. "data/clips" vs. the default
        # absolute CLIPS_DIR) would otherwise hash to a different uuid5 and
        # silently duplicate the same clip instead of upserting over it,
        # breaking the re-run idempotency this ID scheme exists for.
        clip_path = clip_path.resolve()
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
        # Truthy check, not `is not None` -- an empty string must be
        # treated the same as omitted (fall back to the filename-derived
        # source_id), matching search.py's `_filter_conditions()` check on
        # the other side. Found via integration testing: with `is not
        # None`, source_id="" was stored as a literal empty-string
        # source_id at index time, but a search filtering by source_id=""
        # silently matched everything (search.py's falsy check treats ""
        # as "no filter") -- an empty string meant two different things
        # depending which side of the system read it.
        if source_id:
            meta["source_id"] = source_id
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
