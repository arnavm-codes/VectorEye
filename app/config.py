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
