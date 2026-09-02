# VectorEye (POC)

Plug-and-play video library search for CCTV/surveillance footage. Recordings
are chunked into 10-second clips, embedded with CLIP into a shared
text-video vector space, and searched via natural-language query +
similarity search in Qdrant. **No LLM ever watches or reasons about video
content** — retrieval is pure nearest-neighbor over embeddings. Groq (free
tier) is used only to clean up the user's query text and narrate results,
never to look at footage.

Design rationale and decision history live in the vault note (Obsidian
Vault#1, `Projects/VectorEye.md`) — this README covers setup/status
only.

## Stack

- **Chunking**: FFmpeg (fixed 10s segments)
- **Embedding**: CLIP (`open_clip`, `ViT-B-32`/openai weights), sparse
  frame sampling + mean-pool per clip
- **Vector DB**: Qdrant (self-hosted via Docker)
- **API**: FastAPI (`/search` — raw retrieval, `/chat` — Groq-wrapped)
- **Chat**: Groq free-tier API, query-side only

All models are open-weight and run locally/CPU-only. Groq is the one
externally-hosted piece, isolated to the non-critical chat layer.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GROQ_API_KEY if you want the /chat endpoint
docker compose up -d   # starts Qdrant on localhost:6333
```

Place the POC's source videos in `data/raw_videos/` (`.mp4`/`.mov`/`.mkv`/`.avi`).

## Run the pipeline

```bash
uv run python scripts/run_pipeline.py
```

This chunks every video in `data/raw_videos/` into `data/clips/`, embeds
each clip with CLIP, and upserts them into the Qdrant `video_clips`
collection.

## Streamlit demo UI

```bash
uv run streamlit run app/ui/streamlit_app.py
```

Type a query, optionally toggle "Groq chat mode" (requires `GROQ_API_KEY`),
and view the matched clips with inline playback. Talks directly to the
retrieval code — no separate API process needed for the demo.

## Serve the API

```bash
uv run uvicorn app.main:app --reload
```

- `POST /search {"query": "blue car at the gate"}` — raw CLIP+Qdrant
  retrieval, no LLM involved.
- `POST /chat {"query": "..."}` — same retrieval, wrapped with Groq for
  query cleanup + a conversational summary of results.
- `GET /clip/{filename}` — serves a matched clip file for playback.

## Retrieval evaluation (RAGAS)

This system never generates a text answer grounded in video content, so
RAGAS's generation-facing metrics (faithfulness, answer relevancy) don't
apply. What's evaluated instead is retrieval quality itself, via RAGAS's
**ID-based context precision/recall** — pure clip-ID matching against a
hand-labeled ground truth, no LLM judge involved.

1. Fill in `data/eval_queries.json` with real queries and the correct clip
   filenames for each (label these by watching the actual footage).
2. `uv run python -m app.eval.ragas_eval`

Reports per-query and mean context precision/recall.

## Status

Scaffolded, not yet run end-to-end — awaiting the 3 POC source videos.
Progress tracked in the vault note, not here.
