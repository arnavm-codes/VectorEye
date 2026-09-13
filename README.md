```
█   █ █████  ███  █████  ███  ████  █████ █   █ █████   
█░  █░█░░░░░█ ░░░  ░█░░░█ ░░█ █░░░█ █░░░░░ █ █ ░█░░░░░  
█░░ █░████░░█░ ░░░  █░░░█░ ░█░████░░████░░░ █ ░ ████░░░ 
 █░█ ░█░░░░ █░░     █░░ █░░ █░█░░█░ █░░░░   █░ ░█░░░░   
  █ ░ █████░ ███    █░░  ███ ░█░░░█░█████░  █░░ █████░  
   ░ ░ ░░░░░  ░░░    ░░   ░░░ ░░░  ░ ░░░░░   ░░  ░░░░░  
    ░   ░░░░░  ░░░    ░    ░░░  ░   ░ ░░░░░   ░   ░░░░░ 
```


A plug-and-play semantic search engine for **any video library** —
surveillance/CCTV archives, media libraries, sports or event footage,
dashcam/bodycam recordings, personal video collections, anything where you
have a pile of video and want to find the moment matching a description.
Recordings are chunked into short clips, embedded with CLIP into a shared
text-video vector space, and searched via natural-language query +
similarity search in Qdrant. **No LLM ever watches or reasons about video
content** — retrieval is pure nearest-neighbor over embeddings, so it scales
to large libraries without per-clip LLM cost. Groq (free tier) is used only
to clean up the user's query text and narrate results, never to look at
footage.

The system doesn't assume anything domain-specific about the source
footage — clips are grouped under a generic `source_id` (in practice: a
source video, a camera feed, a recording device, or whatever identifier
makes sense for your library) with optional filtering, plus open-ended
`tags` (a flat list of labels) and `attributes` (a structured key/value
dict) for anything else a deployment wants to track — a trip name and
location, a course and lecture number, a camera site — none of it hardcoded
into the pipeline. So the same pipeline works whether you're searching CCTV
recordings for "a red car at the gate" or a personal video archive for "the
dog running on the beach."

## Stack

- **Chunking**: FFmpeg (fixed-length, overlapping segments — configurable)
- **Visual embedding**: CLIP (`open_clip`, `ViT-B-32`/openai weights, or a
  Long-CLIP backend), sparse frame sampling + max-pool per clip
- **Speech-content search (optional, `ENABLE_AUDIO_SEARCH`)**: `faster-whisper`
  (`tiny`, CPU) with its built-in VAD skips clips with no detected speech;
  transcripts are embedded separately with `sentence-transformers`
  (`all-MiniLM-L6-v2`) and fused with the visual ranking at query time via
  Qdrant's native Reciprocal Rank Fusion (`use_transcript_fusion`)
- **Attribute verification (optional, experimental, `ENABLE_ATTRIBUTE_VERIFICATION`)**:
  an open-vocabulary object detector (`ultralytics` YOLO-World) plus a spaCy
  dependency parse fix CLIP's weak attribute-object binding (e.g. "blue car"
  scored as "blue" + "car" independently rather than as one bound concept)
- **On-screen text search (optional, `ENABLE_OCR_SEARCH`)**: Tesseract
  (`pytesseract`, a system binary — no ML framework, no GPU/CUDA risk) reads
  slide/whiteboard/signage text from a sparse frame sample; embedded with the
  same `sentence-transformers` model as transcripts and fused the same way
  (`use_ocr_fusion`)
- **Vector DB**: Qdrant (self-hosted via Docker)
- **API**: FastAPI (`/search` — raw retrieval, `/chat` — Groq-wrapped,
  `/ingest` / `/ingest/batch` / `/watch` — ingestion, `/sources` /
  `/retention/sweep` — visibility and cleanup)
- **Chat**: Groq free-tier API, query-side only
- **Model backends are swappable without touching call sites**: every
  model-selection point in this codebase (visual embedding backend,
  Whisper size, sentence-transformer model, YOLO-World checkpoint, OCR
  engine) is either a single config value (swapping checkpoints/sizes) or a
  config value behind one dispatch function with one loader per engine
  family (swapping libraries entirely, e.g. Tesseract → PaddleOCR) — see
  each pipeline module's `_load_*` functions.

## Architecture

![VectorEye architecture](docs/assets/architecture.png)

All models are open-weight and run locally/CPU-only. Groq is the one
externally-hosted piece, isolated to the non-critical chat layer. Nothing
about the pipeline is CCTV-specific — swap in whatever video library you
have and it works the same way.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GROQ_API_KEY if you want the /chat endpoint
docker compose up -d   # starts Qdrant on localhost:6333
```

Place the source videos in `data/raw_videos/` (`.mp4`/`.mov`/`.mkv`/`.avi`).

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

Type a query and view the matched clips with inline playback. Talks
directly to the retrieval code — no separate API process needed for the
demo. The sidebar has a toggle for each optional feature (see "Stack"
above; each disabled until its `ENABLE_*` flag is on and the collection has
been re-indexed with it): "Groq chat mode" (requires `GROQ_API_KEY`), "Fuse
in speech content", "Fuse in on-screen text", and "Verify attributes" — the
last three are independent and combinable.

## How to use the API

This section walks through setup and every endpoint with copy-pasteable
`curl` commands. Everything below assumes Qdrant is running
(`docker compose up -d`) and the API server is up:

```bash
uv sync
cp .env.example .env          # fill in GROQ_API_KEY if you want /chat
docker compose up -d          # starts Qdrant on localhost:6333
uv run uvicorn app.main:app --reload   # starts the API on localhost:8000
```

Check it's alive:

```bash
curl localhost:8000/health
# {"status": "ok"}
```

### 1. Get video into the system (ingestion)

Unlike the old `scripts/run_pipeline.py` CLI script (still works, for a
one-time bulk load from `data/raw_videos/`), the API lets you add video
one at a time, in bulk, or continuously — without re-processing what's
already indexed.

**One video at a time** — point at any video file already on the server's
disk, give it a `source_id` (whatever grouping makes sense for your
library: a camera name, a trip, a course), and optionally `tags`/`attributes`:

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{
  "video_path": "/path/to/video.mp4",
  "source_id": "front-gate-camera",
  "tags": ["site:hq"],
  "attributes": {"location": "Kyoto"}
}'
# {"job_id": "..."}
```

Chunking + embedding takes real time (seconds to minutes depending on
video length), so this returns immediately with a `job_id`. Poll it for
progress:

```bash
curl localhost:8000/ingest/<job_id>
# {"status": "embedding_indexing", ...}   -> then "done" or "failed"
```

A video that's already been ingested and hasn't changed since is
automatically **skipped** (status `"skipped"`) instead of re-processed —
add `"force": true` to the request body to re-process it anyway.

**A whole folder of videos at once** — for a static archive (a batch of
teaching videos, an existing folder of trip footage). Point at a directory
and every video in it gets its own ingest job, each with its own
filename-derived `source_id` unless you give one explicitly per file:

```bash
curl -X POST localhost:8000/ingest/batch -H "Content-Type: application/json" -d '{
  "directory": "/path/to/videos",
  "tags": ["archive:2024"]
}'
# {"batch_id": "...", "job_ids": [...]}

curl localhost:8000/ingest/batch/<batch_id>
# shows every job's status plus one overall status: "running" / "done" / "failed"
```

(Or pass an explicit `"videos": [{"video_path": "...", "source_id": "..."}, ...]`
list instead of `"directory"` if each video needs its own metadata.)

**A folder that keeps getting new files** — for a live source like a CCTV
NVR export directory. Register it once and it's watched automatically:

```bash
curl -X POST localhost:8000/watch -H "Content-Type: application/json" -d '{
  "directory": "/path/to/nvr-export",
  "tags": ["camera:lobby"],
  "interval_seconds": 30
}'
# {"watch_id": "..."}
```

Every 30 seconds it checks the folder for new or changed files and
auto-ingests them — already-processed, unchanged files are skipped, so
re-checking a mostly-unchanged folder is cheap.

```bash
curl localhost:8000/watch/<watch_id>       # status + list of jobs it triggered
curl localhost:8000/watch                  # list every registered watch
curl -X DELETE localhost:8000/watch/<watch_id>   # stop watching
```

### 2. Search

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{
  "query": "blue car at the gate",
  "top_k": 5
}'
```

Raw CLIP+Qdrant retrieval, no LLM involved. Narrow it down with
`"source_id": "front-gate-camera"` (match one source), or a more general
`"filters"` list for anything in `tags`/`attributes`:

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{
  "query": "blue car at the gate",
  "filters": [
    {"field": "tags", "op": "eq", "value": "site:hq"},
    {"field": "attributes.location", "op": "in", "value": ["Kyoto", "Osaka"]}
  ]
}'
```

`op` is `"eq"` (exact match) or `"in"` (matches any value in a list).

Fold in speech content or on-screen text (both opt-in, both need indexing
with `ENABLE_AUDIO_SEARCH`/`ENABLE_OCR_SEARCH` on — see "Stack" above — to
have anything to fuse in) with `"use_transcript_fusion": true` and/or
`"use_ocr_fusion": true` — combinable together for a 3-way fusion:

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{
  "query": "what did the sign say at the gate",
  "use_ocr_fusion": true
}'
```

Want a conversational answer instead of raw hits? Use `/chat` (needs
`GROQ_API_KEY` in `.env`) — same retrieval underneath, just narrated:

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{
  "query": "did anyone show up at the gate last night"
}'
```

Play back a matched clip (each search result includes its `clip_path`;
just the filename is what this endpoint wants):

```bash
curl localhost:8000/clip/front-gate-camera_clip000010.mp4 --output clip.mp4
```

### 3. See what's indexed, and clean it up

```bash
curl localhost:8000/sources               # every source_id, with clip counts
curl localhost:8000/sources/<source_id>   # detail for one source
curl -X DELETE localhost:8000/sources/<source_id>   # delete it (Qdrant + disk)
```

For a source that should auto-expire (a CCTV rolling window), set
`retention_days` in its `attributes` at ingest time — anything past that
age gets deleted automatically if you turn on the background sweep
(`ENABLE_RETENTION_SWEEP=true` in `.env`), or trigger a sweep manually any
time:

```bash
curl -X POST localhost:8000/retention/sweep
# {"deleted": 12}
```

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

## License

MIT — see [LICENSE](LICENSE).

