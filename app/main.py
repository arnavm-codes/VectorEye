"""FastAPI entrypoint: search endpoint (+ optional Groq-explained chat endpoint).

Run: uv run uvicorn app.main:app --reload
"""

from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.search import search_clips
from app.config import (
    ENABLE_RETENTION_SWEEP,
    GROQ_API_KEY,
    RETENTION_SWEEP_INTERVAL_SECONDS,
    WATCH_POLL_INTERVAL_SECONDS,
)
from app.pipeline import ingest as ingest_pipeline
from app.pipeline import retention as retention_pipeline
from app.pipeline import sources as sources_pipeline
from app.pipeline import watch as watch_pipeline
from app.pipeline.chunker import VIDEO_EXTENSIONS

app = FastAPI(title="Video Library RAG (POC)")


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
    top_k: int = 5
    source_id: str | None = None
    filters: list[FilterCondition] | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestRequest(BaseModel):
    # A path already reachable on the server's filesystem -- e.g. dropped
    # there by an NVR export, a synced folder, or a prior upload step.
    # Direct multipart file upload isn't supported yet (a real gap, not
    # silently assumed away -- left for whenever a caller actually needs to
    # hand over raw bytes instead of a path the server can already read).
    video_path: str
    # None falls back to index_clips()'s legacy filename-parsing convention
    # (see app.pipeline.ingest.create_job) -- lets a batch directory-scan
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
    directory: str | None = None
    # Shared tags/attributes applied to every video found when scanning
    # `directory` (ignored when `videos` is given -- each entry already
    # carries its own).
    tags: list[str] = []
    attributes: dict[str, Any] = {}
    force: bool = False


class WatchRequest(BaseModel):
    directory: str
    source_id: str | None = None
    tags: list[str] = []
    attributes: dict[str, Any] = {}
    interval_seconds: int = WATCH_POLL_INTERVAL_SECONDS


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    """Raw retrieval: no LLM involved, pure CLIP + Qdrant similarity search."""
    filters = [f.model_dump() for f in req.filters] if req.filters else None
    return {"results": search_clips(req.query, top_k=req.top_k, source_id=req.source_id, filters=filters)}


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


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Chunks, embeds, and indexes one video, tracked as a background job
    (chunking+embedding a video is real wall-clock work, not a
    request-response-speed operation -- see app.pipeline.ingest). Returns
    immediately with a job_id; poll GET /ingest/{job_id} for progress."""
    video_path = Path(req.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail=f"video_path does not exist or is not a file: {req.video_path}")

    job_id = ingest_pipeline.create_job(video_path, req.source_id, req.tags, req.attributes)
    background_tasks.add_task(
        ingest_pipeline.run_job, job_id, video_path, req.source_id, req.tags, req.attributes, req.force
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
    """One-shot bulk ingest for a static archive: either an explicit list
    of videos (each with its own metadata), or a directory to scan for
    every video file in it. Fans out to one per-video job each (same
    machinery as POST /ingest), grouped under one batch_id -- this is the
    static-archive counterpart to scripts/run_pipeline.py's chunk_all(), as
    an API rather than a CLI script."""
    if (req.videos is None) == (req.directory is None):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'videos' or 'directory'")

    entries: list[tuple[Path, str | None, list[str], dict]]
    if req.directory is not None:
        directory = Path(req.directory)
        if not directory.is_dir():
            raise HTTPException(
                status_code=400, detail=f"directory does not exist or is not a directory: {req.directory}"
            )
        video_paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_paths:
            raise HTTPException(status_code=400, detail=f"no video files found in {req.directory}")
        # source_id=None per video -- each gets its own filename-derived
        # source_id (see IngestRequest.source_id), not one value shared
        # across the whole directory.
        entries = [(p, None, req.tags, req.attributes) for p in video_paths]
    else:
        entries = []
        for v in req.videos:
            video_path = Path(v.video_path)
            if not video_path.is_file():
                raise HTTPException(
                    status_code=400, detail=f"video_path does not exist or is not a file: {v.video_path}"
                )
            entries.append((video_path, v.source_id, v.tags, v.attributes))

    job_ids = []
    for video_path, source_id, tags, attributes in entries:
        job_id = ingest_pipeline.create_job(video_path, source_id, tags, attributes)
        background_tasks.add_task(
            ingest_pipeline.run_job, job_id, video_path, source_id, tags, attributes, req.force
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
    """Registers `directory` to be polled on `interval_seconds` for new or
    changed video files (see app.pipeline.watch), each auto-ingested via
    the same per-video job machinery as POST /ingest -- this is what lets a
    continuously-updating source (e.g. a CCTV export directory) stay
    indexed without anyone re-running an ingest call by hand. `async def`
    deliberately (not the usual sync handler) so this runs on the event
    loop thread, required for asyncio.create_task() inside start_watch()."""
    directory = Path(req.directory)
    if not directory.is_dir():
        raise HTTPException(
            status_code=400, detail=f"directory does not exist or is not a directory: {req.directory}"
        )
    watch_id = watch_pipeline.start_watch(
        directory, req.source_id, req.tags, req.attributes, req.interval_seconds
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


@app.get("/sources/{source_id}")
def get_source(source_id: str):
    source = sources_pipeline.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    return source


@app.delete("/sources/{source_id}")
def delete_source(source_id: str):
    """Deletes every clip for `source_id`, both from Qdrant and on disk.
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


@app.get("/clip/{clip_filename}")
def get_clip(clip_filename: str):
    from app.config import CLIPS_DIR

    return FileResponse(CLIPS_DIR / clip_filename)
