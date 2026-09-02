"""FastAPI entrypoint: search endpoint (+ optional Groq-explained chat endpoint).

Run: uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.search import search_clips
from app.config import GROQ_API_KEY

app = FastAPI(title="Video Library RAG (POC)")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    camera_id: str | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.get("/clip/{clip_filename}")
def get_clip(clip_filename: str):
    from app.config import CLIPS_DIR

    return FileResponse(CLIPS_DIR / clip_filename)
