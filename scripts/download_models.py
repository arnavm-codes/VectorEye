"""Downloads every model's weights up front, so nothing is fetched mid-pipeline or on the first search.

Called by setup.sh / setup.bat. Idempotent: anything already cached is skipped or served from cache. Safe to
re-run, e.g. after changing WHISPER_MODEL_SIZE or YOLO_WORLD_MODEL in .env.

Usage: uv run python scripts/download_models.py

Models fetched (all of them, regardless of which optional features are switched on in .env):
  1. Long-CLIP-B checkpoint  -> data/model_cache/longclip-B.pt       (~600 MB, default embedding backend)
  2. OpenAI CLIP ViT-B-32    -> Hugging Face cache                   (~350 MB, EMBEDDING_BACKEND=clip; also reused by
                                                                      attribute verification, if enabled)
  3. faster-whisper model    -> Hugging Face cache                   (size = WHISPER_MODEL_SIZE, ~75 MB for "tiny")
  4. all-MiniLM-L6-v2        -> Hugging Face cache                   (~90 MB, transcript embeddings)
  5. YOLO-World + its text encoder -> project root                   (attribute verification, ~90 MB + ~350 MB)
The Silero VAD used by faster-whisper ships inside the package, and Tesseract (if you use the OCR branch) is a
system binary, not a model download.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Ultralytics saves bare-filename weights (yolov8l-worldv2.pt) relative to the current directory, and the app loads
# them the same way, so downloading must happen from the project root or they would never be found at runtime.
os.chdir(PROJECT_ROOT)

from app.config import (  # noqa: E402
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    LONGCLIP_CHECKPOINT_PATH,
    TRANSCRIPT_EMBED_MODEL,
    WHISPER_MODEL_SIZE,
    YOLO_WORLD_MODEL,
)

LONGCLIP_REPO = "BeichenZhang/LongCLIP-B"  # the checkpoint is published as longclip-B.pt in this Hugging Face repo


def download_longclip() -> str:
    if LONGCLIP_CHECKPOINT_PATH.exists():
        return f"already present ({LONGCLIP_CHECKPOINT_PATH})"
    from huggingface_hub import hf_hub_download

    LONGCLIP_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(LONGCLIP_REPO, LONGCLIP_CHECKPOINT_PATH.name, local_dir=LONGCLIP_CHECKPOINT_PATH.parent)
    return f"saved to {LONGCLIP_CHECKPOINT_PATH}"


def download_open_clip() -> str:
    import open_clip

    open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device="cpu")
    return f"{CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) cached"


def download_whisper() -> str:
    from faster_whisper.utils import download_model

    download_model(WHISPER_MODEL_SIZE)
    return f"faster-whisper '{WHISPER_MODEL_SIZE}' cached"


def download_sentence_model() -> str:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(TRANSCRIPT_EMBED_MODEL, device="cpu")
    return f"{TRANSCRIPT_EMBED_MODEL} cached"


def download_yolo_world() -> str:
    from ultralytics import YOLOWorld

    detector = YOLOWorld(YOLO_WORLD_MODEL)
    # The detector's CLIP text encoder is only fetched the first time classes are set, so trigger that here too --
    # otherwise the first attribute-verified search would stall on a ~350 MB download.
    detector.set_classes(["object"])
    return f"{YOLO_WORLD_MODEL} + text encoder cached"


STEPS = [
    ("Long-CLIP-B checkpoint", download_longclip),
    ("OpenAI CLIP (open_clip)", download_open_clip),
    ("faster-whisper speech model", download_whisper),
    ("sentence-transformer (transcripts)", download_sentence_model),
    ("YOLO-World detector", download_yolo_world),
]


def main() -> int:
    failures = []
    for i, (name, fn) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {name} ...", flush=True)
        try:
            print(f"      ok: {fn()}", flush=True)
        except Exception as exc:  # keep going so one bad download doesn't hide the others
            failures.append(name)
            print(f"      FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}. Check your network and re-run "
              f"(completed downloads are cached and won't repeat).", file=sys.stderr)
        return 1
    print("\nAll model weights are downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
