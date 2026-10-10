"""Remove the legacy pre-cut clips bucket ("chunks").

VectorEye used to store every 10 s clip in a second S3 bucket; clips are now cut from the raw
videos on demand, so that bucket is dead weight. This deletes its objects and then the bucket.

Usage:
    uv run python scripts/cleanup_chunks.py                  # dry run: report only
    uv run python scripts/cleanup_chunks.py --apply          # delete objects, then the bucket
    uv run python scripts/cleanup_chunks.py --bucket other   # if yours isn't called "chunks"

Run it only after you've checked that search results play correctly through /clip. Raw videos
(the raw-videos bucket) are never touched.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botocore.exceptions import ClientError

from app import storage
from app.config import RAW_VIDEOS_BUCKET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="chunks")
    parser.add_argument("--apply", action="store_true", help="delete (default is a dry run)")
    args = parser.parse_args()
    if args.bucket == RAW_VIDEOS_BUCKET:
        sys.exit(f"refusing: '{args.bucket}' is the raw-videos bucket")

    s3 = storage.get_s3()
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError:
        print(f"Bucket '{args.bucket}' does not exist; nothing to do.")
        return

    keys = storage.list_keys(args.bucket)
    size = sum(o["Size"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket) for o in page.get("Contents", []))
    print(f"Bucket '{args.bucket}': {len(keys)} objects, {size / 1e6:.1f} MB")
    if not args.apply:
        print("Dry run -- nothing deleted. Re-run with --apply to delete the objects and the bucket.")
        return
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=args.bucket, Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True})
    s3.delete_bucket(Bucket=args.bucket)
    print(f"Deleted {len(keys)} objects and bucket '{args.bucket}'.")


if __name__ == "__main__":
    main()
