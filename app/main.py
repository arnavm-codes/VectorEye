"""FastAPI entrypoint: search endpoint (+ optional Groq-explained chat endpoint).

Run: uv run uvicorn app.main:app --reload
"""

import threading
import uuid
from pathlib import Path

from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import storage
from app.api.search import search_clips
from app.config import GROQ_API_KEY

app = FastAPI(title="VectorEye API")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)
    camera_id: str | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1)


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
    return {"results": search_clips(req.query, top_k=req.top_k, camera_id=req.camera_id)}


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
    video_key: str | None = None   # None = process every video in raw-videos-cctv
    reindex_all: bool = False      # True = re-embed clips even if already indexed


_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def _run_index_job(job_id: str, req: IndexRequest):
    from app.pipeline.chunker import chunk_all
    from app.pipeline.indexer import index_clips

    try:
        _jobs[job_id]["status"] = "chunking"
        clips = chunk_all(video_key=req.video_key)
        _jobs[job_id].update(status="indexing", clips_created=len(clips))
        video_stem = Path(req.video_key).stem if req.video_key else None
        indexed = index_clips(only_new=not req.reindex_all, video_stem=video_stem)
        _jobs[job_id].update(status="done", clips_indexed=indexed)
    except Exception as exc:  # noqa: BLE001
        _jobs[job_id].update(status="error", error=str(exc))
    finally:
        _job_lock.release()


@app.post("/index", status_code=202)
def start_index(req: IndexRequest, background: BackgroundTasks):
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


@app.get("/clip/{clip_filename}")
def get_clip(clip_filename: str):
    try:
        obj = storage.open_clip_stream(clip_filename)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            raise HTTPException(status_code=404, detail="Clip not found")
        raise
    return StreamingResponse(obj["Body"].iter_chunks(chunk_size=65536), media_type="video/mp4")
