"""Create the S3 buckets VectorEye needs (raw videos + clips) if they're missing.

Usage: uv run python scripts/ensure_buckets.py
Safe to re-run: existing buckets are left alone. Uses the S3_* / *_BUCKET
settings from .env, so it works against Floci or any S3-compatible store.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage
from app.config import CLIPS_BUCKET, RAW_VIDEOS_BUCKET


def main():
    created = storage.ensure_buckets()
    for bucket in (RAW_VIDEOS_BUCKET, CLIPS_BUCKET):
        print(f"  {bucket}: {'created' if bucket in created else 'already exists'}")


if __name__ == "__main__":
    main()
