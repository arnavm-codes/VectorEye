"""File-fingerprint change detection: lets ingestion skip a video that
hasn't changed since it was last processed, instead of re-chunking and
re-embedding it every time it's seen again -- the "wasteful full rescan"
problem flagged in the original plug-and-play audit (see vault note entry,
2026-09-13), and the specific gap that makes a watch-folder worker (Phase
4) practical instead of wasteful by construction.

Fingerprint is (mtime, size) -- cheap to read (a stat() call, not reading
the video's actual bytes), the same heuristic tools like rsync/make use to
detect "did this file change" without hashing gigabytes of video. Not a
content hash: a file replaced with different content that happens to land
on the same mtime+size would be missed. Accepted tradeoff at this
project's scale/rigor level (consistent with e.g. MIN_SIMILARITY_SCORE's
empirical-not-exhaustive derivation) -- a real gap worth knowing about, not
silently assumed away.

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
from pathlib import Path

from app.config import PROJECT_ROOT

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


def _fingerprint(video_path: Path) -> list:
    st = video_path.stat()
    return [st.st_mtime, st.st_size]


def has_changed(video_path: Path) -> bool:
    """True if `video_path` is new, or its (mtime, size) differs from the
    last time `mark_processed` was called for it."""
    state = _load_state()
    key = str(video_path.resolve())
    return state.get(key) != _fingerprint(video_path)


def mark_processed(video_path: Path) -> None:
    """Records `video_path`'s current fingerprint as processed. Call only
    after successfully chunking+indexing it -- marking a failed attempt as
    processed would make `has_changed` wrongly skip it next time."""
    state = _load_state()
    key = str(video_path.resolve())
    state[key] = _fingerprint(video_path)
    _save_state(state)
