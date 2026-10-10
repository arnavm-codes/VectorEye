"""Central configuration: paths, model/backend selection, and tuned constants.

Every value here is read from an environment variable (via .env) where it's
meant to be deployment-tunable, or hardcoded where it's a project-specific
constant derived empirically (see inline comments and the vault note for
how each was chosen).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# --- Object storage (Floci S3 locally; any S3-compatible store in deployment) ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:4566")
# Endpoint embedded in presigned URLs -- must be reachable by whoever plays the clip.
S3_PUBLIC_ENDPOINT = os.environ.get("S3_PUBLIC_ENDPOINT", S3_ENDPOINT)
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "test")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "test")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_VERIFY_SSL = os.environ.get("S3_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
RAW_VIDEOS_BUCKET = os.environ.get("RAW_VIDEOS_BUCKET", "raw-videos")
CLIPS_BUCKET = os.environ.get("CLIPS_BUCKET", "chunks")
PRESIGN_EXPIRY_SECONDS = int(os.environ.get("PRESIGN_EXPIRY_SECONDS", "3600"))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

CLIP_DURATION_SECONDS = 10
# A second, offset chunking pass (start times CHUNK_OVERLAP_SECONDS, +CLIP_DURATION_SECONDS,
# ...) runs alongside the base 0/10/20s pass, so an event spanning a fixed-chunk
# boundary still lands cleanly inside at least one clip instead of being split
# across two weak-scoring ones. 5s = 50% overlap with a 10s clip duration.
CHUNK_OVERLAP_SECONDS = 5
FRAMES_PER_CLIP = 10  # denser sampling improves color/attribute-binding accuracy
# (e.g. "a red car" vs a similar blue car); 3 was too sparse -- see vault
# note's retrieval-quality findings. Negligible cost at this POC's scale.

CLIP_MODEL_NAME = "ViT-B-32-quickgelu"  # matches OpenAI's original weights exactly
CLIP_PRETRAINED = "openai"

# Embedding backend under test on this branch (test/model-long-clip):
# "clip" = baseline open_clip model above; "longclip" = vendored Long-CLIP
# (see vault note's model-comparison table for why -- targets CLIP's 77-token
# truncation / compositional attribute-binding weakness, e.g. "red car" vs a
# similar blue car). Each backend writes to its own Qdrant collection so the
# baseline data isn't touched while comparing.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "longclip")
LONGCLIP_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "model_cache" / "longclip-B.pt"
LONGCLIP_EMBED_DIM = 512  # LongCLIP-B is ViT-B/16-based, same dim as baseline

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Fails loudly on an unrecognized EMBEDDING_BACKEND instead of silently
# falling back to the default -- found as a real gap during the OCR
# backend-architecture planning (see vault note "OCR effort kicked off"
# entry, 2026-09-13): the old code (`if == "longclip": ... else: ...`)
# treated any unrecognized value, including a typo, as "clip" with no
# error. This is now the house convention for every model with a
# family-switching backend (see app.pipeline.embedder._load_model() and
# app.pipeline.ocr's dispatch, which both fail the same way).
if EMBEDDING_BACKEND == "longclip":
    QDRANT_COLLECTION = "video_clips_longclip"
    CLIP_EMBED_DIM = LONGCLIP_EMBED_DIM
elif EMBEDDING_BACKEND == "clip":
    QDRANT_COLLECTION = "video_clips"
    CLIP_EMBED_DIM = 512  # ViT-B-32 output dim
else:
    raise ValueError(f"Unrecognized EMBEDDING_BACKEND {EMBEDDING_BACKEND!r} (expected 'clip' or 'longclip')")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Minimum Cosine score for a search hit to count as a real match rather than
# noise. Derived empirically (see vault note "Suggested deeper-testing
# measures" / math-verification section, 2026-09-02): 50 true-positive vs 66
# false-positive queries against two confirmed-clean reference clips, true
# scores ~0.294 mean, false scores ~0.195 mean, threshold chosen to maximize
# TPR-FPR (Youden's J) -> TPR=100%, FPR=3% at this cutoff. Only validated
# against two scene types (car-at-gate, mall-food-court) -- treat as a
# starting point, not a universally-tuned constant, until tested against
# more varied footage.
MIN_SIMILARITY_SCORE = float(os.environ.get("MIN_SIMILARITY_SCORE", "0.237"))

# Speech-content search: VAD-gated Whisper transcription of each clip's
# audio, embedded separately from the visual CLIP vector and fused with it
# at query time (see vault note "Feasibility study" entry, 2026-09-03).
# Disableable since it adds real per-clip indexing cost (transcription) that
# not every deployment needs.
ENABLE_AUDIO_SEARCH = os.environ.get("ENABLE_AUDIO_SEARCH", "true").lower() == "true"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
# Below this many characters, a transcript is treated as VAD noise/silence
# rather than real speech (Whisper can hallucinate a short filler phrase on
# near-silent audio even with vad_filter=True).
MIN_TRANSCRIPT_CHARS = 3

# Separate from CLIP: CLIP's text encoder is tuned for image-caption
# alignment, not general sentence semantics, so transcript text needs its
# own embedding model/vector space (confirmed in the feasibility study).
TRANSCRIPT_EMBED_MODEL = "all-MiniLM-L6-v2"
TRANSCRIPT_EMBED_DIM = 384

# Score threshold for the transcript branch of fusion search, gating out weak
# candidates the same way MIN_SIMILARITY_SCORE does for the visual branch.
# Derived 2026-09-09 by probing real query scores against this project's
# speech-bearing clips: true matches scored 0.29-0.66, off-topic/false
# matches (including a control query with no speech relevance) scored
# 0.005-0.18, with a boundary-case transcript fragment at 0.163 -- 0.2 sits
# in the gap. Same caveat as MIN_SIMILARITY_SCORE: validated on this
# project's two source videos only, not a universally-tuned constant.
MIN_TRANSCRIPT_SCORE = float(os.environ.get("MIN_TRANSCRIPT_SCORE", "0.2"))

# Attribute-binding fix (see vault note "Attribute-binding / bag-of-words
# retrieval failure", 2026-09-08): CLIP's global embedding doesn't reliably
# bind an attribute (e.g. "blue") to the object it describes (e.g. "car")
# once compared cross-modally, so a query like "blue car" can be scored as
# "contains blue" + "contains car" independently. Rather than trying to
# repair CLIP's embedding, an open-vocabulary detector (YOLO-World, chosen
# over Grounding DINO/OWLv2/YOLOE for its speed/size/accuracy balance on
# common-object categories -- see vault note's comparison table) localizes
# the object noun first, then the attribute is verified with CLIP only on
# that crop, where there's just one concept in play. Only runs on the
# candidate pool that already cleared the normal CLIP shortlist, since it's
# real per-clip cost (frame extraction + detection + a CLIP call per box).
ENABLE_ATTRIBUTE_VERIFICATION = (
    os.environ.get("ENABLE_ATTRIBUTE_VERIFICATION", "false").lower() == "true"
)
# "l" (large) -- slowest of the three benchmarked sizes (see vault note,
# 2026-09-08), but real manual UI testing found M's result quality better
# than S's, so L is being tried next to see if quality keeps improving with
# size (at a further latency cost) before settling on a default.
YOLO_WORLD_MODEL = os.environ.get("YOLO_WORLD_MODEL", "yolov8l-worldv2.pt")
# How many top visual-search hits get re-ranked by attribute verification.
# Larger = more thorough but more per-query detector/CLIP calls.
ATTRIBUTE_VERIFICATION_POOL = int(os.environ.get("ATTRIBUTE_VERIFICATION_POOL", "10"))
# Frames sampled per clip *for this step only* -- deliberately sparser than
# FRAMES_PER_CLIP (used at indexing time to build a clip's search embedding):
# verification only needs to confirm the object/attribute appears somewhere
# in the clip, not characterize the whole clip the way the indexed embedding
# does, and this runs at query time (real per-query latency) rather than
# once at indexing time. See vault note's per-frame benchmark for why this
# was the larger lever versus detector size alone.
ATTRIBUTE_VERIFICATION_FRAMES = int(os.environ.get("ATTRIBUTE_VERIFICATION_FRAMES", "3"))

# Generic metadata layer (see vault note "Plug-and-play audit" entry,
# 2026-09-13): every clip payload carries a required `source_id` (the
# generic grouping key -- a camera, a course, a trip, whatever a given
# deployment's videos are grouped by) plus two open-ended buckets: `tags`
# (a flat list of free-form strings, e.g. "trip:japan-2024") and
# `attributes` (a structured key/value dict, e.g. {"location": "Kyoto"}).
# Neither bucket's keys are hardcoded in the pipeline -- a deployment just
# starts passing whatever tags/attributes it wants at ingestion time.
#
# PAYLOAD_INDEX_FIELDS lists which top-level payload fields get a Qdrant
# payload index (required for that field to be filtered/matched
# efficiently at query time -- see qdrant_client.create_payload_index calls
# in app/pipeline/indexer.py). `source_id` is indexed by default since
# every deployment filters by it; a deployment that wants fast filtering on
# a specific attribute (e.g. "attributes.location" for the personal-gallery
# case, "attributes.course_id" for teaching video) adds it here via env var
# rather than editing code. This is the "per-deployment schema
# declaration" -- deliberately just a list of field paths, not a bigger
# schema/type system, since Qdrant's payload is already schemaless JSON and
# the only thing actually required up front is which fields need an index.
PAYLOAD_INDEX_FIELDS = [
    f.strip()
    for f in os.environ.get("PAYLOAD_INDEX_FIELDS", "source_id,tags").split(",")
    if f.strip()
]

# Default polling interval for a watch-folder worker (Phase 4 of the
# plug-and-play effort, see vault note "Plug-and-play audit" entry,
# 2026-09-13) -- how often it re-scans its directory for new/changed
# videos. Overridable per-watch via POST /watch's interval_seconds.
WATCH_POLL_INTERVAL_SECONDS = int(os.environ.get("WATCH_POLL_INTERVAL_SECONDS", "30"))

# Retention sweep (Phase 5 of the plug-and-play effort, same vault note
# entry): periodically deletes clips whose `attributes.retention_days` has
# elapsed since indexing -- the CCTV-style rolling-window case. Off by
# default -- this is a destructive, automatic operation, so it needs an
# explicit deployment opt-in; a deployment that never sets retention_days
# on anything is unaffected either way, but the automatic loop itself
# shouldn't run without someone deciding to turn it on. POST
# /retention/sweep (a manual, explicit trigger) is always available
# regardless of this flag.
ENABLE_RETENTION_SWEEP = os.environ.get("ENABLE_RETENTION_SWEEP", "false").lower() == "true"
RETENTION_SWEEP_INTERVAL_SECONDS = int(os.environ.get("RETENTION_SWEEP_INTERVAL_SECONDS", "3600"))

# On-screen text search (OCR) -- same additive/config-gated/off-by-default
# pattern as speech-content search (ENABLE_AUDIO_SEARCH above): an
# independent named vector fused in via RRF at query time, aimed mainly at
# slide/whiteboard-heavy content (see vault note "OCR effort kicked off"
# entry, 2026-09-13, for the engine comparison behind this default).
ENABLE_OCR_SEARCH = os.environ.get("ENABLE_OCR_SEARCH", "false").lower() == "true"
# "tesseract" (default) -- Apache 2.0, no ML-framework dependency at all,
# sidesteps the torch/CUDA-pull risk category entirely rather than needing
# another careful CPU-only pin. Documented fallback if real-footage testing
# shows this isn't accurate enough for small/distorted CCTV-style text:
# "paddleocr" (Apache 2.0, still lightweight at its small tier, better
# scene-text accuracy, at the cost of a second ML framework to vet) -- not
# implemented yet, addable later as a pure addition with zero call-site
# changes (see app.pipeline.ocr's backend dispatch).
OCR_BACKEND = os.environ.get("OCR_BACKEND", "tesseract")
# Frames sampled per clip for OCR -- separate from FRAMES_PER_CLIP (used
# for the visual embedding): on-screen text is typically static across many
# consecutive frames (a slide held for several seconds), so a sparser
# sample is enough to catch it, same reasoning as
# ATTRIBUTE_VERIFICATION_FRAMES's sparser sampling for that feature.
OCR_FRAMES_PER_CLIP = int(os.environ.get("OCR_FRAMES_PER_CLIP", "3"))
# Same role as MIN_TRANSCRIPT_SCORE for the transcript fusion branch --
# gates weak/off-topic ocr_text matches out of the fused ranking. Starting
# at the same 0.2 value as MIN_TRANSCRIPT_SCORE since both branches embed
# with the same sentence-transformer model into the same kind of vector
# space -- explicitly a starting point pending real calibration against
# real OCR'd text, same caveat as every other empirically-derived threshold
# in this file.
MIN_OCR_SCORE = float(os.environ.get("MIN_OCR_SCORE", "0.2"))
