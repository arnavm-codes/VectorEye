"""File-fingerprint change detection: lets ingestion skip a video that
hasn't changed since it was last processed, instead of re-chunking and
re-embedding it every time it's seen again -- the "wasteful full rescan"
problem flagged in the original plug-and-play audit (see vault note entry,
2026-09-13), and the specific gap that makes a watch-folder worker (Phase
4) practical instead of wasteful by construction.

Fingerprint is the S3 object's (ETag, size) -- cheap to read (a HEAD request,
not downloading the video), and the ETag changes whenever the object's content
does. (For multipart uploads the ETag is not a plain content hash, but it is
still stable for an unchanged object.)

Persisted to a small JSON file (unlike the in-memory-only job/batch
registries in app.pipeline.ingest) since change detection's whole point is
to stay useful across a long-running watch process and its restarts -- an
in-memory-only version would silently reprocess every video after every
restart, defeating the purpose. Read-modify-write on every call, no file
locking -- fine for this project's expected concurrency (a handful of
ingest jobs at a time, not a high-throughput multi-worker deployment); a
real concurrent-writer scenario would need proper locking or a real DB
instead of a shared JSON file.
"""

import json
from app import storage
from app.config import PROJECT_ROOT, RAW_VIDEOS_BUCKET

_STATE_PATH = PROJECT_ROOT / "data" / ".ingest_state.json"


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state))


def _state_key(video_key: str) -> str:
    return f"s3://{RAW_VIDEOS_BUCKET}/{video_key}"


def has_changed(video_key: str) -> bool:
    """True if the raw video `video_key` is new, or its (ETag, size) differs
    from the last time `mark_processed` was called for it."""
    return _load_state().get(_state_key(video_key)) != storage.raw_video_fingerprint(video_key)


def mark_processed(video_key: str) -> None:
    """Records `video_key`'s current fingerprint as processed. Call only
    after successfully chunking+indexing it -- marking a failed attempt as
    processed would make `has_changed` wrongly skip it next time."""
    state = _load_state()
    state[_state_key(video_key)] = storage.raw_video_fingerprint(video_key)
    _save_state(state)
