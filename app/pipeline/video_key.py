"""Maps a stored clip key back to the raw video it was cut from.

Needed because points indexed before `video_key` existed carry only a clip key
(`<raw video key minus extension>_clip<start_ts>.mp4`), and the raw video's
extension isn't recoverable from that alone -- it has to be looked up against
the raw-videos bucket. Used by the indexer (bulk path, where no explicit
video_key is passed) and by scripts/backfill_video_key.py.
"""

import re
from pathlib import PurePosixPath

_CLIP_KEY_RE = re.compile(r"^(?P<stem>.+)_clip\d+$")


def clip_video_stem(clip_key: str) -> str | None:
    """`cam1/clip_clip000010.mp4` -> `cam1/clip`. None if the key doesn't
    follow the clip naming convention."""
    m = _CLIP_KEY_RE.match(PurePosixPath(clip_key).with_suffix("").as_posix())
    return m.group("stem") if m else None


def build_stem_index(raw_keys: list[str]) -> dict[str, list[str]]:
    """Raw video keys grouped by extension-less key. More than one entry for a
    stem means two raw videos (e.g. a.mp4 and a.mkv) would have produced the
    same clip names, so the clip's origin is ambiguous."""
    index: dict[str, list[str]] = {}
    for key in raw_keys:
        index.setdefault(PurePosixPath(key).with_suffix("").as_posix(), []).append(key)
    return index


def resolve_video_key(clip_key: str, stem_index: dict[str, list[str]]) -> tuple[str | None, str]:
    """Returns (video_key, status), status one of "ok", "unparseable",
    "missing" (no raw video with that stem) or "ambiguous"."""
    stem = clip_video_stem(clip_key)
    if stem is None:
        return None, "unparseable"
    candidates = stem_index.get(stem, [])
    if not candidates:
        return None, "missing"
    if len(candidates) > 1:
        return None, "ambiguous"
    return candidates[0], "ok"
