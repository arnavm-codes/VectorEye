"""FastAPI entrypoint: search endpoint (+ optional Groq-explained chat endpoint).

Run: uv run uvicorn app.main:app --reload
"""

import threading
import uuid
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import storage
from app.api.search import search_clips
from app.config import (
    ENABLE_RETENTION_SWEEP,
    GROQ_API_KEY,
    RETENTION_SWEEP_INTERVAL_SECONDS,
    WATCH_POLL_INTERVAL_SECONDS,
)
from app.pipeline import clip_service
from app.pipeline import ingest as ingest_pipeline
from app.pipeline import retention as retention_pipeline
from app.pipeline import sources as sources_pipeline
from app.pipeline import watch as watch_pipeline

app = FastAPI(title="VectorEye API")


@app.on_event("startup")
async def _check_clip_signing():
    # Fail at boot, not on the first search: clip URLs can't be signed without the secret.
    clip_service.check_configured()


@app.on_event("startup")
async def _start_retention_sweep():
    if ENABLE_RETENTION_SWEEP:
        import asyncio

        asyncio.create_task(retention_pipeline.sweep_loop(RETENTION_SWEEP_INTERVAL_SECONDS))


class FilterCondition(BaseModel):
    """One generic payload filter: {field, op, value}. `field` is a payload
    key, dotted for nested attributes (e.g. "attributes.location"). `op`
    must stay in sync with app.api.search._SUPPORTED_FILTER_OPS -- kept as
    a Literal (not a plain str) so an unsupported op is rejected by
    pydantic with a 422 at the request boundary, instead of reaching
    search_clips() and raising an uncaught ValueError (a 500)."""

    field: str
    op: Literal["eq", "in"]
    value: Any


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)
    source_id: str | None = None
    filters: list[FilterCondition] | None = None
    # Fuse in transcribed speech content / OCR'd on-screen text (see
    # app.api.search_clips). Off by default -- neither is purely additive
    # to a visual-only ranking, so both stay opt-in.
    use_transcript_fusion: bool = False
    use_ocr_fusion: bool = False


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)


class IngestRequest(BaseModel):
    # Object key of a video already in the raw-videos bucket. Direct
    # multipart upload isn't supported (a real gap, not silently assumed
    # away) -- put the file in the bucket first.
    video_key: str
    # None falls back to a source_id derived from the video's key
    # (see app.pipeline.ingest.create_job) -- lets a batch prefix-scan
    # ingest each video under its own filename-derived source_id instead of
    # forcing one shared value onto every video in the batch.
    source_id: str | None = None
    tags: list[str] = []
    attributes: dict[str, Any] = {}
    # Bypasses change detection (app.pipeline.change_detection) -- normally
    # a video that looks unchanged since its last successful ingest is
    # skipped; force=True re-processes it anyway.
    force: bool = False


class IngestBatchRequest(BaseModel):
    # Exactly one of these two must be given.
    videos: list[IngestRequest] | None = None
    # Key prefix to scan in the raw-videos bucket ("" or omitted = all videos).
    prefix: str | None = None
    # Shared tags/attributes applied to every video found when scanning
    # `prefix` (ignored when `videos` is given -- each entry already
    # carries its own).
    tags: list[str] = []
    attributes: dict[str, Any] = {}
    force: bool = False


class WatchRequest(BaseModel):
    prefix: str = ""
    source_id: str | None = None
    tags: list[str] = []
    attributes: dict[str, Any] = {}
    interval_seconds: int = WATCH_POLL_INTERVAL_SECONDS


@app.get("/health")
def health():
    from app.pipeline.indexer import get_client

    s3_ok, s3_msg = storage.s3_healthy()
    try:
        get_client().get_collections()
        qdrant_ok, qdrant_msg = True, "ok"
    except Exception as exc:  # noqa: BLE001
        qdrant_ok, qdrant_msg = False, str(exc)
    return {
        "status": "ok" if (s3_ok and qdrant_ok) else "degraded",
        "s3": s3_msg,
        "qdrant": qdrant_msg,
    }


@app.post("/search")
def search(req: SearchRequest):
    """Raw retrieval: no LLM involved, pure CLIP + Qdrant similarity search."""
    filters = [f.model_dump() for f in req.filters] if req.filters else None
    return {
        "results": search_clips(
            req.query,
            top_k=req.top_k,
            source_id=req.source_id,
            filters=filters,
            use_transcript_fusion=req.use_transcript_fusion,
            use_ocr_fusion=req.use_ocr_fusion,
        )
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Same retrieval, wrapped with Groq for query cleanup + conversational summary."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set; use /search instead."}

    from app.chat.groq_wrapper import clean_query, explain_results

    cleaned = clean_query(req.query)
    results = search_clips(cleaned, top_k=req.top_k)
    summary = explain_results(req.query, results)
    return {"cleaned_query": cleaned, "results": results, "summary": summary}


class IndexRequest(BaseModel):
    video_key: str | None = None   # None = process every video in raw-videos
    reindex_all: bool = False      # True = re-process videos even if unchanged since their last index


_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def _run_index_job(job_id: str, req: IndexRequest):
    """Bulk (re)index: runs the same per-video job as /ingest for every raw video. Unchanged
    videos are skipped unless `reindex_all`; a re-indexed video keeps the source_id, tags and
    attributes it was originally ingested with."""
    job = _jobs[job_id]
    try:
        video_keys = [req.video_key] if req.video_key else storage.list_raw_videos()
        job.update(status="indexing", videos_total=len(video_keys), videos_indexed=0,
                   videos_skipped=0, clips_indexed=0, errors=[])
        for key in video_keys:
            sub_id = ingest_pipeline.create_job(key, None, None, None)
            ingest_pipeline.run_job(sub_id, key, None, None, None, force=req.reindex_all)
            result = ingest_pipeline.get_job(sub_id)
            if result["status"] == "done":
                job["videos_indexed"] += 1
                job["clips_indexed"] += result["clips_indexed"]
            elif result["status"] == "skipped":
                job["videos_skipped"] += 1
            else:
                job["errors"].append({"video_key": key, "error": result["error"]})
        job["status"] = "error" if job["errors"] else "done"
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", error=str(exc))
    finally:
        _job_lock.release()


@app.post("/index", status_code=202)
def start_index(req: IndexRequest, background: BackgroundTasks):
    if req.video_key and not storage.raw_video_exists(req.video_key):
        raise HTTPException(status_code=404, detail=f"video_key not found in raw-videos bucket: {req.video_key}")
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An indexing job is already running.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    background.add_task(_run_index_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/index/{job_id}")
def index_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **_jobs[job_id]}


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Chunks, embeds, and indexes one raw video (an object key in the
    raw-videos bucket), tracked as a background job (chunking+embedding a
    video is real wall-clock work, not a request-response-speed operation --
    see app.pipeline.ingest). Returns immediately with a job_id; poll
    GET /ingest/{job_id} for progress."""
    if not storage.raw_video_exists(req.video_key):
        raise HTTPException(status_code=400, detail=f"video_key not found in raw-videos bucket: {req.video_key}")

    job_id = ingest_pipeline.create_job(req.video_key, req.source_id, req.tags, req.attributes)
    background_tasks.add_task(
        ingest_pipeline.run_job, job_id, req.video_key, req.source_id, req.tags, req.attributes, req.force
    )
    return {"job_id": job_id}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str):
    job = ingest_pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/ingest/batch")
def ingest_batch(req: IngestBatchRequest, background_tasks: BackgroundTasks):
    """One-shot bulk ingest for a static archive: either an explicit list of
    raw videos (each with its own metadata), or a key prefix to scan in the
    raw-videos bucket ("" = every video). Fans out to one per-video job each
    (same machinery as POST /ingest), grouped under one batch_id."""
    if req.videos is not None and req.prefix is not None:
        raise HTTPException(status_code=400, detail="Provide at most one of 'videos' or 'prefix'")

    entries: list[tuple[str, str | None, list[str], dict]]
    if req.videos is None:
        # source_id=None per video -- each gets its own filename-derived
        # source_id (see IngestRequest.source_id), not one value shared
        # across the whole prefix.
        video_keys = storage.list_raw_videos(req.prefix or "")
        if not video_keys:
            raise HTTPException(status_code=400, detail=f"no video files found under prefix {req.prefix or ''!r}")
        entries = [(k, None, req.tags, req.attributes) for k in video_keys]
    else:
        entries = []
        for v in req.videos:
            if not storage.raw_video_exists(v.video_key):
                raise HTTPException(
                    status_code=400, detail=f"video_key not found in raw-videos bucket: {v.video_key}"
                )
            entries.append((v.video_key, v.source_id, v.tags, v.attributes))

    job_ids = []
    for video_key, source_id, tags, attributes in entries:
        job_id = ingest_pipeline.create_job(video_key, source_id, tags, attributes)
        background_tasks.add_task(
            ingest_pipeline.run_job, job_id, video_key, source_id, tags, attributes, req.force
        )
        job_ids.append(job_id)

    batch_id = ingest_pipeline.create_batch(job_ids)
    return {"batch_id": batch_id, "job_ids": job_ids}


@app.get("/ingest/batch/{batch_id}")
def ingest_batch_status(batch_id: str):
    batch = ingest_pipeline.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


@app.post("/watch")
async def watch(req: WatchRequest):
    """Registers a raw-videos bucket key prefix to be polled on
    `interval_seconds` for new or changed videos (see app.pipeline.watch),
    each auto-ingested via the same per-video job machinery as POST /ingest.
    `async def` deliberately (not the usual sync handler) so this runs on the
    event loop thread, required for asyncio.create_task() inside start_watch()."""
    watch_id = watch_pipeline.start_watch(
        req.prefix, req.source_id, req.tags, req.attributes, req.interval_seconds
    )
    return {"watch_id": watch_id}


@app.get("/watch")
def list_watches():
    return {"watches": watch_pipeline.list_watches()}


@app.get("/watch/{watch_id}")
def watch_status(watch_id: str):
    w = watch_pipeline.get_watch(watch_id)
    if w is None:
        raise HTTPException(status_code=404, detail="watch not found")
    return w


@app.delete("/watch/{watch_id}")
def stop_watch(watch_id: str):
    if not watch_pipeline.stop_watch(watch_id):
        raise HTTPException(status_code=404, detail="watch not found")
    return {"status": "stopping"}


@app.get("/sources")
def list_sources():
    """Lists every indexed source_id with a summary (clip count, tag union,
    first/last indexed_at) -- see app.pipeline.sources. Necessarily scans
    the whole collection, so cheap at this project's scale, not meant to
    be called on a hot path at a much larger one."""
    return {"sources": sources_pipeline.list_sources()}


@app.get("/sources/{source_id:path}")
def get_source(source_id: str):
    source = sources_pipeline.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    return source


@app.delete("/sources/{source_id:path}")
def delete_source(source_id: str):
    """Deletes every clip for `source_id`, both from Qdrant and the clips bucket.
    See app.pipeline.sources.delete_source for the known gap around
    change-detection state not being cleared for the underlying raw
    video(s)."""
    deleted = sources_pipeline.delete_source(source_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="source not found")
    return {"deleted": deleted}


@app.post("/retention/sweep")
def retention_sweep():
    """Manually triggers one retention sweep pass (see
    app.pipeline.retention) -- deletes any clip whose attributes.retention_days
    has elapsed since indexing. Always available regardless of
    ENABLE_RETENTION_SWEEP, which only gates the *automatic* periodic loop;
    an explicit one-shot trigger doesn't need that same opt-in."""
    return {"deleted": retention_pipeline.sweep_once()}


@app.get("/clip")
def get_clip_window(video_key: str, start: int, end: int, exp: int, sig: str):
    """Dynamic clip: cuts [start, end) out of the raw video on demand (cached). Only
    reachable through the signed URLs search returns -- see app.pipeline.clip_service."""
    if not clip_service.verify_signature(video_key, start, end, exp, sig):
        raise HTTPException(status_code=403, detail="invalid or expired clip URL")
    try:
        path = clip_service.get_clip_file(video_key, start, end)
    except clip_service.ClipInvalid as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except clip_service.ClipNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except clip_service.ClipError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})
