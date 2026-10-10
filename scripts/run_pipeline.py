"""One-shot POC pipeline: chunk raw videos -> embed -> index into Qdrant.

Usage: uv run python scripts/run_pipeline.py [--reindex-all]
Prereq: Qdrant running (docker compose up -d), Floci/S3 reachable, videos in the
raw-videos bucket. Clips are uploaded to the chunks bucket; incremental by
default (--reindex-all re-embeds clips that are already indexed).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.chunker import chunk_all
from app.pipeline.indexer import index_clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex-all", action="store_true", help="re-embed clips even if already indexed")
    args = parser.parse_args()

    print("=== Step 1: chunking raw videos ===")
    clips = chunk_all()
    if not clips:
        print("No videos found in the raw-videos bucket. Upload videos there and re-run.")
        return

    print("\n=== Step 2: embedding + indexing into Qdrant ===")
    index_clips(only_new=not args.reindex_all)


if __name__ == "__main__":
    main()
