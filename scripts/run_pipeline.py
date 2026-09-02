"""One-shot POC pipeline: chunk raw videos -> embed -> index into Qdrant.

Usage: uv run python scripts/run_pipeline.py
Prereq: Qdrant running (docker compose up -d), videos placed in data/raw_videos/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.chunker import chunk_all
from app.pipeline.indexer import index_clips


def main():
    print("=== Step 1: chunking raw videos ===")
    clips = chunk_all()
    if not clips:
        print("No videos found in data/raw_videos/. Add the 3 POC videos there and re-run.")
        return

    print("\n=== Step 2: embedding + indexing into Qdrant ===")
    index_clips()


if __name__ == "__main__":
    main()
