"""Retention sweep: deletes clips whose source has aged past its
`retention_days` -- Phase 5 of the plug-and-play effort (see vault note
"Plug-and-play audit" entry, 2026-09-13), the CCTV-style rolling-window
case the original audit flagged as missing.

`retention_days` lives in a clip's `attributes` dict (Phase 0's generic
bucket), not a dedicated payload field -- consistent with everything else
in that bucket being deployment-supplied metadata rather than a hardcoded
schema field. A clip with no `retention_days` attribute is never touched
by the sweep, so this is a no-op for every deployment that doesn't opt in.

The automatic periodic loop (`sweep_loop`) is gated behind
app.config.ENABLE_RETENTION_SWEEP -- off by default, since this deletes
data automatically. `sweep_once()` itself (a manual, explicit trigger via
POST /retention/sweep) has no such gate, matching DELETE /sources/{id}'s
existing precedent: an explicit, one-shot destructive action doesn't need
a separate opt-in flag the way an unattended recurring one does.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import QDRANT_COLLECTION
from app.pipeline.indexer import get_client


def sweep_once() -> int:
    """Deletes every clip whose `attributes.retention_days` has elapsed
    since its `indexed_at`. Returns the number of clips deleted."""
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return 0

    now = datetime.now(timezone.utc)
    to_delete = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            retention_days = p.payload.get("attributes", {}).get("retention_days")
            indexed_at = p.payload.get("indexed_at")
            if retention_days is None or not indexed_at:
                continue
            try:
                indexed_dt = datetime.fromisoformat(indexed_at)
            except ValueError:
                continue
            if now - indexed_dt > timedelta(days=retention_days):
                to_delete.append(p)
        if offset is None:
            break

    if not to_delete:
        return 0

    client.delete(collection_name=QDRANT_COLLECTION, points_selector=[p.id for p in to_delete])
    for p in to_delete:
        clip_path = p.payload.get("clip_path")
        if clip_path:
            Path(clip_path).unlink(missing_ok=True)
    return len(to_delete)


async def sweep_loop(interval_seconds: int) -> None:
    """Runs sweep_once() forever on an interval. Started as a background
    asyncio task at app startup when ENABLE_RETENTION_SWEEP is set (see
    app.main's startup handler)."""
    while True:
        deleted = sweep_once()
        if deleted:
            print(f"Retention sweep: deleted {deleted} expired clip(s).")
        await asyncio.sleep(interval_seconds)
