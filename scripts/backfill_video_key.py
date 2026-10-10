"""Backfill `video_key` (the raw video a clip was cut from) onto already-indexed points.

Points indexed before `video_key` existed carry only `clip_path`. This resolves each
one against the raw-videos bucket and sets the payload field in place -- no
re-embedding, point IDs and vectors untouched.

Usage:
    uv run python scripts/backfill_video_key.py            # dry run: report only
    uv run python scripts/backfill_video_key.py --apply    # write the payloads

Idempotent: points that already have `video_key` are skipped. Points whose raw
video is missing from the bucket, or ambiguous (a.mp4 and a.mkv both present),
are listed and left alone.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PayloadSchemaType

from app import storage
from app.config import QDRANT_COLLECTION
from app.pipeline.indexer import get_client
from app.pipeline.video_key import build_stem_index, resolve_video_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the payloads (default is a dry run)")
    args = parser.parse_args()

    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        print(f"Collection '{QDRANT_COLLECTION}' does not exist; nothing to do.")
        return

    stem_index = build_stem_index(storage.list_raw_videos())
    by_video: dict[str, list] = defaultdict(list)
    problems: dict[str, list[str]] = defaultdict(list)
    already = total = 0

    offset = None
    while True:
        points, offset = client.scroll(QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True)
        for p in points:
            total += 1
            if p.payload.get("video_key"):
                already += 1
                continue
            clip_key = p.payload.get("clip_path")
            video_key, status = resolve_video_key(clip_key, stem_index) if clip_key else (None, "unparseable")
            if video_key:
                by_video[video_key].append(p.id)
            else:
                problems[status].append(str(clip_key))
        if offset is None:
            break

    resolvable = sum(len(ids) for ids in by_video.values())
    print(f"{total} points: {already} already have video_key, {resolvable} resolvable, "
          f"{sum(len(v) for v in problems.values())} unresolved")
    for video_key, ids in sorted(by_video.items()):
        print(f"  {video_key}: {len(ids)} points")
    for status, keys in problems.items():
        print(f"  {status}: {len(keys)} points, e.g. {keys[:3]}")

    if not args.apply:
        print("\nDry run -- nothing written. Re-run with --apply to set video_key.")
        return

    # The index is what P3's delete-by-video_key relies on; creating it is idempotent.
    client.create_payload_index(QDRANT_COLLECTION, "video_key", PayloadSchemaType.KEYWORD)
    for video_key, ids in by_video.items():
        client.set_payload(QDRANT_COLLECTION, payload={"video_key": video_key}, points=ids)
    print(f"\nApplied: video_key set on {resolvable} points.")
    if problems:
        print("Unresolved points were left unchanged (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
