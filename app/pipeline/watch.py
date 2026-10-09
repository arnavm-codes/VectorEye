"""Watch-folder worker for continuously-updating sources (e.g. a CCTV
NVR's export directory) -- Phase 4 of the plug-and-play effort (see vault
note "Plug-and-play audit" entry, 2026-09-13).

Polls a registered raw-videos bucket prefix on an interval and fires an ingest job (via
app.pipeline.ingest) for any video file that's new or changed since it was
last processed. Relies entirely on app.pipeline.change_detection to skip
everything else -- without that, re-polling a directory would mean
re-chunking and re-embedding every video in it on every poll, which is
exactly the "wasteful full rescan" problem the plug-and-play audit flagged
for dynamic sources. This module has no change-detection logic of its own;
it's just what calls has_changed()/run_job() on a schedule.

Runs as an asyncio background task inside the FastAPI process (started via
POST /watch, which must be an `async def` handler so it runs on the event
loop thread -- asyncio.create_task requires a running loop, which a sync
FastAPI handler doesn't have since those run in a worker thread). Stopped
via DELETE /watch/{watch_id}, which cancels the task.

In-memory registry only, same scope caveat as app.pipeline.ingest's job/
batch registries: doesn't survive a process restart, and a real multi-
worker deployment would need every watch registered on every worker (or a
single dedicated watcher process) rather than assuming one process owns
all watches. Not addressed here -- consistent with the rest of this
project's POC-scale persistence decisions.
"""

import asyncio
import uuid

from app import storage
from app.pipeline.change_detection import has_changed
from app.pipeline.ingest import create_job, run_job

_WATCHES: dict[str, dict] = {}


async def _watch_loop(
    watch_id: str, prefix: str, source_id: str | None, tags: list[str], attributes: dict, interval_seconds: int
) -> None:
    state = _WATCHES[watch_id]
    try:
        while True:
            video_keys = await asyncio.to_thread(storage.list_raw_videos, prefix)
            for video_key in video_keys:
                if not await asyncio.to_thread(has_changed, video_key):
                    continue
                job_id = create_job(video_key, source_id, tags, attributes)
                state["jobs_triggered"].append(job_id)
                # Awaited (via to_thread, since run_job is sync/CPU-bound)
                # rather than fired concurrently -- one video ingested at a
                # time per watch, so the watcher can't pile up dozens of
                # simultaneous chunk/embed jobs if a directory suddenly
                # gains many files at once; the next poll picks up anything
                # still waiting.
                await asyncio.to_thread(run_job, job_id, video_key, source_id, tags, attributes)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        state["status"] = "stopped"
        raise


def start_watch(
    prefix: str,
    source_id: str | None,
    tags: list[str],
    attributes: dict,
    interval_seconds: int,
) -> str:
    watch_id = str(uuid.uuid4())
    _WATCHES[watch_id] = {
        "watch_id": watch_id,
        "prefix": prefix,
        "source_id": source_id,
        "interval_seconds": interval_seconds,
        "status": "running",
        "jobs_triggered": [],
    }
    task = asyncio.create_task(
        _watch_loop(watch_id, prefix, source_id, tags, attributes, interval_seconds)
    )
    _WATCHES[watch_id]["_task"] = task
    return watch_id


def stop_watch(watch_id: str) -> bool:
    entry = _WATCHES.get(watch_id)
    if entry is None:
        return False
    entry["_task"].cancel()
    return True


def get_watch(watch_id: str) -> dict | None:
    entry = _WATCHES.get(watch_id)
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if k != "_task"}


def list_watches() -> list[dict]:
    return [get_watch(wid) for wid in _WATCHES]
