import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"

CLIP_DURATION_SECONDS = 10
FRAMES_PER_CLIP = 10  # denser sampling improves color/attribute-binding accuracy
# (e.g. "a red car" vs a similar blue car); 3 was too sparse -- see vault
# note's retrieval-quality findings. Negligible cost at this POC's scale.

CLIP_MODEL_NAME = "ViT-B-32-quickgelu"  # matches OpenAI's original weights exactly
CLIP_PRETRAINED = "openai"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "video_clips"
CLIP_EMBED_DIM = 512  # ViT-B-32 output dim

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
