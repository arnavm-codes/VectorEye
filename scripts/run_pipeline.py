"""Bulk pipeline: index every video in the raw-videos bucket into Qdrant.

Usage: uv run python scripts/run_pipeline.py [--reindex-all]
Prereq: Qdrant running (docker compose up -d), Floci/S3 reachable, videos in the
raw-videos bucket. Each video is cut into temp clips, embedded and indexed (nothing is
stored besides the Qdrant points). Videos unchanged since their last index are skipped;
--reindex-all re-processes them anyway. Equivalent to POST /index.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage
from app.pipeline import ingest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex-all", action="store_true")
    args = parser.parse_args()

    video_keys = storage.list_raw_videos()
    if not video_keys:
        print("No videos found in the raw-videos bucket. Upload videos there and re-run.")
        return
    for key in video_keys:
        job_id = ingest.create_job(key, None, None, None)
        ingest.run_job(job_id, key, None, None, None, force=args.reindex_all)
        job = ingest.get_job(job_id)
        print(f"{key}: {job['status']}" + (f" ({job['clips_indexed']} clips)" if job["status"] == "done" else f" -- {job['error']}" if job["status"] == "failed" else ""))


if __name__ == "__main__":
    main()
