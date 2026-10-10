"""Visibility into what's indexed, plus source deletion -- Phase 5 of the
plug-and-play effort (see vault note "Plug-and-play audit" entry,
2026-09-13). Before this, there was no way to see what's indexed short of
querying Qdrant directly, and no way to remove a source's data short of the
ad-hoc client.delete() calls used throughout this session's manual test
cleanup -- this module is that cleanup logic promoted to a real API.
"""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import QDRANT_COLLECTION
from app.pipeline.change_detection import forget
from app.pipeline.indexer import get_client


def _scroll_all(client, scroll_filter: Filter | None = None) -> list:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _source_filter(source_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])


def list_sources() -> list[dict]:
    """Aggregates every indexed source_id into a summary: clip count, tag
    union, earliest/latest indexed_at. Necessarily scans the whole
    collection client-side (there's no way to know which source_ids exist
    without looking at every point) -- fine at this project's POC scale,
    would want a server-side aggregation for a much larger collection."""
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return []

    summaries: dict[str, dict] = {}
    for p in _scroll_all(client):
        sid = p.payload.get("source_id")
        if sid is None:
            continue
        s = summaries.setdefault(
            sid,
            {"source_id": sid, "clip_count": 0, "tags": set(), "first_indexed_at": None, "last_indexed_at": None},
        )
        s["clip_count"] += 1
        s["tags"].update(p.payload.get("tags", []))
        ts = p.payload.get("indexed_at")
        if ts:
            if s["first_indexed_at"] is None or ts < s["first_indexed_at"]:
                s["first_indexed_at"] = ts
            if s["last_indexed_at"] is None or ts > s["last_indexed_at"]:
                s["last_indexed_at"] = ts

    result = []
    for s in summaries.values():
        s["tags"] = sorted(s["tags"])
        result.append(s)
    return sorted(result, key=lambda s: s["source_id"])


def get_source(source_id: str) -> dict | None:
    """One source's detail: the same summary fields as list_sources() plus
    its attributes and clip_paths (capped -- a response detail, not a
    pagination API). Filters server-side via Qdrant's scroll_filter, unlike
    list_sources() which has no choice but to scan everything."""
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return None

    points = _scroll_all(client, scroll_filter=_source_filter(source_id))
    if not points:
        return None

    tags = set()
    attributes = {}
    timestamps = []
    clip_paths = []
    video_keys = set()
    for p in points:
        tags.update(p.payload.get("tags", []))
        attributes.update(p.payload.get("attributes", {}))
        ts = p.payload.get("indexed_at")
        if ts:
            timestamps.append(ts)
        clip_paths.append(p.payload.get("clip_path"))
        video_keys.add(p.payload.get("video_key"))

    return {
        "source_id": source_id,
        "clip_count": len(points),
        "tags": sorted(tags),
        "attributes": attributes,
        "first_indexed_at": min(timestamps) if timestamps else None,
        "last_indexed_at": max(timestamps) if timestamps else None,
        "video_keys": sorted(k for k in video_keys if k),
        "clip_paths": sorted(clip_paths)[:100],
    }


def delete_source(source_id: str) -> int:
    """Deletes every point for `source_id` from Qdrant and returns how many were removed
    (0 if the source doesn't exist). Raw videos are never touched -- they are the only copy
    of the footage -- and there are no stored clips to delete.

    Also resets change detection for the raw video(s) behind those points, so ingesting the
    same unchanged video again re-indexes it instead of being skipped as "already done"."""
    client = get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return 0

    points = _scroll_all(client, scroll_filter=_source_filter(source_id))
    if not points:
        return 0

    client.delete(collection_name=QDRANT_COLLECTION, points_selector=_source_filter(source_id))
    for video_key in {p.payload.get("video_key") for p in points} - {None}:
        forget(video_key)
    return len(points)
