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

RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"

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

if EMBEDDING_BACKEND == "longclip":
    QDRANT_COLLECTION = "video_clips_longclip"
    CLIP_EMBED_DIM = LONGCLIP_EMBED_DIM
else:
    QDRANT_COLLECTION = "video_clips"
    CLIP_EMBED_DIM = 512  # ViT-B-32 output dim

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
