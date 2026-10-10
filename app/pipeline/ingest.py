"""Async single-video ingestion: cut -> embed -> index, tracked via an
in-memory job registry. Status: queued -> indexing -> done | failed | skipped.

Cutting+embedding a video takes real wall-clock time (see vault note's
performance measurements, 2026-09-03), so POST /ingest (app/main.py)
returns a job_id immediately and runs the actual work in a FastAPI
background task -- this module owns the job state and the work itself,
main.py just wires the HTTP layer to it.

This is the per-video ingestion path (Phase 2 of the plug-and-play effort,
see vault note "Plug-and-play audit" entry, 2026-09-13): index_video()
operates on exactly the one video being ingested -- so ingesting a new
video doesn't re-embed any clip already indexed for another. Clips are cut
into a temp dir and discarded; nothing is stored in S3.

Also owns the batch registry (Phase 3) -- POST /ingest/batch (app/main.py)
fans out to one job per video via create_job()/run_job() below, then groups
the resulting job_ids under one batch_id via create_batch(), so a static
archive load (many videos, one request) is just N of this same per-video
job rather than a separate code path.

In-memory registries only -- no persistence across a process restart.
Fine at this project's current scale (a single dev-local FastAPI process,
POC-sized job volume); a real deployment running multiple worker processes
would need a shared store (Redis, a DB row) instead of a module-level dict,
since each worker would otherwise see a different, incomplete set of jobs.
"""

import uuid

from app.pipeline.change_detection import has_changed, mark_processed
from app.pipeline.indexer import index_video

_JOBS: dict[str, dict] = {}
_BATCHES: dict[str, list[str]] = {}


def create_job(video_key: str, source_id: str | None, tags: list[str] | None, attributes: dict | None) -> str:
    """Registers a new job in "queued" state and returns its id. Callers
    should schedule `run_job` with the same arguments right after.

    `source_id=None` is a legitimate value, not an omission -- it tells
    index_video() (via run_job) to use the source_id the video already has,
    or else one derived from its key. This is what a prefix-scan batch ingest
    uses, since each video under the prefix should get its own source_id,
    not one value shared across every video in the batch. `tags`/`attributes`
    left as None are likewise inherited from the video's existing points."""
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "video_key": video_key,
        "source_id": source_id,
        "clips_indexed": None,
        "error": None,
    }
    return job_id


def run_job(
    job_id: str,
    video_key: str,
    source_id: str | None,
    tags: list[str] | None,
    attributes: dict | None,
    force: bool = False,
) -> None:
    """Does the actual cut -> embed -> index work for one video, updating
    the job's status as it progresses. Meant to run as a background task,
    not called directly from a request handler (would block the response).

    Skips the work entirely (status "skipped") if `video_key` hasn't
    changed since it was last successfully processed (see
    app.pipeline.change_detection) -- this is what makes a watch-folder
    worker (Phase 4) re-polling its directory cheap instead of wastefully
    re-cutting+re-embedding every video on every poll. `force=True`
    bypasses the check for an explicit "re-ingest this even though it looks
    unchanged" request.
    """
    job = _JOBS[job_id]
    if not force and not has_changed(video_key):
        job["status"] = "skipped"
        job["clips_indexed"] = 0
        return
    try:
        job["status"] = "indexing"
        # Clips are cut into a temp dir, embedded, and discarded; the video's points are
        # swapped in once everything has embedded (see index_video).
        clips_indexed = index_video(video_key, source_id=source_id, tags=tags, attributes=attributes)

        # Marked only after a fully successful run -- a failed attempt must
        # stay eligible for retry next time has_changed() is checked.
        mark_processed(video_key)

        job["status"] = "done"
        job["clips_indexed"] = clips_indexed
    except Exception as exc:
        # Caught broadly and stored on the job rather than left to propagate
        # -- this runs in a background task, so an uncaught exception here
        # would just vanish (no request left to surface it to); recording it
        # on the job is what makes GET /ingest/{job_id} actually useful for
        # a failed run instead of it silently staying "indexing"
        # forever.
        job["status"] = "failed"
        job["error"] = str(exc)


def get_job(job_id: str) -> dict | None:
    return _JOBS.get(job_id)


_TERMINAL_STATUSES = {"done", "failed", "skipped"}


def create_batch(job_ids: list[str]) -> str:
    batch_id = str(uuid.uuid4())
    _BATCHES[batch_id] = job_ids
    return batch_id


def get_batch(batch_id: str) -> dict | None:
    """Returns the batch's job_ids plus each job's current status, and an
    overall status computed live from them (not stored separately, so it
    can never drift out of sync with the underlying jobs): "running" while
    any job hasn't reached a terminal state, else "done" if every job
    succeeded or "failed" if any job failed."""
    job_ids = _BATCHES.get(batch_id)
    if job_ids is None:
        return None
    jobs = [get_job(jid) for jid in job_ids]
    statuses = [j["status"] for j in jobs]
    if not all(s in _TERMINAL_STATUSES for s in statuses):
        overall = "running"
    elif any(s == "failed" for s in statuses):
        overall = "failed"
    else:
        overall = "done"
    return {"batch_id": batch_id, "status": overall, "jobs": jobs}
