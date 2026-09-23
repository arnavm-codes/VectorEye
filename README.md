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

| Layer | Tech | Opt-in? | Config |
|---|---|---|---|
| Chunking | FFmpeg — fixed-length, overlapping segments | always on | configurable |
| Visual embedding | CLIP (`open_clip`, `ViT-B-32`/openai, or Long-CLIP) — sparse frame sampling + max-pool | always on | `EMBEDDING_BACKEND` |
| Speech-content search | `faster-whisper` (`tiny`, CPU, built-in VAD) + `sentence-transformers` (`all-MiniLM-L6-v2`), fused via Qdrant's native RRF | opt-in | `ENABLE_AUDIO_SEARCH`, `use_transcript_fusion` |
| On-screen text search (OCR) | Tesseract (`pytesseract`, system binary — no ML framework, no GPU/CUDA risk) + same sentence-transformer as above | opt-in | `ENABLE_OCR_SEARCH`, `use_ocr_fusion` |
| Date/time filtering | Reuses Tesseract to OCR each clip's first frame for a burned-in NVR/DVR timestamp overlay, regex + `dateparser` extraction, stored as a range-queryable `recorded_at` field — own flag, independent of on-screen-text search | opt-in | `ENABLE_TIMESTAMP_EXTRACTION`, `TIMESTAMP_OCR_ASSUMED_TZ`, `filters: [{"op": "range", ...}]` |
| Attribute verification | Open-vocab object detector (`ultralytics` YOLO-World) + spaCy dependency parse — fixes CLIP's weak attribute-object binding (e.g. "blue car" scored as "blue" + "car" independently) | opt-in, experimental | `ENABLE_ATTRIBUTE_VERIFICATION` |
| Vector DB | Qdrant, self-hosted via Docker | always on | — |
| API | FastAPI — `/search`, `/chat`, `/ingest`, `/ingest/batch`, `/watch`, `/sources`, `/retention/sweep` | always on | — |
| Chat | Groq free-tier API, query-side only | opt-in | `GROQ_API_KEY` |

Every model-selection point above is either a single config value (swapping
checkpoints/sizes within a library) or a config value behind one dispatch
function with one loader per engine family (swapping libraries entirely,
e.g. Tesseract → PaddleOCR) — see each pipeline module's `_load_*`
functions. This means every backend in the table can be swapped without
touching any call site.

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
    {"field": "attributes.location", "op": "in", "value": ["Kyoto", "Osaka"]},
    {"field": "recorded_at", "op": "range", "value": {"gte": "2026-09-15T00:00:00Z", "lte": "2026-09-15T23:59:59Z"}}
  ]
}'
```

`op` is `"eq"` (exact match), `"in"` (matches any value in a list), or `"range"` (a `{"gte", "lte"}` window, either bound
optional — currently only meaningful for `recorded_at`, the one datetime-indexed field).

**Narrowing by date/time (`recorded_at`):** for CCTV-style deployments with a continuously growing multi-camera
library, `source_id` alone isn't enough to keep a search scoped — you also want to say "clips from yesterday
afternoon on camera 3" rather than searching the whole history. Set `ENABLE_TIMESTAMP_EXTRACTION=true` and each
clip's first frame is OCR'd for a burned-in NVR/DVR timestamp overlay (common on real camera exports), stored as a
range-queryable `recorded_at` field. A caller-supplied `attributes.recorded_at` at ingest time always takes
precedence over OCR and skips the OCR call entirely. Sources without a burned-in overlay (or with OCR off) simply
have no `recorded_at` and won't match a `range` filter — not every clip needs one for the feature to be useful.
`TIMESTAMP_OCR_ASSUMED_TZ` (default `UTC`) sets the timezone assumed for a naive timestamp (no offset in the overlay
or the manual value) before it's stored as UTC.

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

