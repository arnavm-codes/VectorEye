"""Best-effort extraction of a burned-in NVR/DVR timestamp overlay from a
clip's first frame, plus normalization of any recorded_at value (OCR'd or
caller-supplied) into a UTC ISO 8601 string for storage.

Why OCR, not video-container metadata: app.pipeline.chunker re-encodes
every clip (no `-map_metadata`), which drops container-level fields like
`creation_time` -- a timestamp baked into the visible pixels survives
re-encoding untouched, where container metadata does not. See vault note
"Date/time search filters" entry for the fuller reasoning.

Full-frame regex scan (not a configurable per-source crop region): zero
setup cost across any source, at the cost of being noisier than a
known-region crop would be. Requiring a date immediately followed by a
time in the same match (not a bare date alone) is what keeps false
positives from unrelated on-screen numbers (a sign, a price) low enough
for this to be worth doing without per-camera configuration.
"""

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import dateparser
from PIL import Image

from app.config import TIMESTAMP_OCR_ASSUMED_TZ
from app.pipeline.ocr import extract_text

# Date immediately followed (same match) by a time -- both required, so a
# bare number or a lone date mention elsewhere in the frame's on-screen
# text doesn't get mistaken for a recording timestamp. Covers the common
# NVR/DVR overlay orderings: YYYY-MM-DD / YYYY/MM/DD (ISO-ish) and
# MM/DD/YYYY / DD-MM-YYYY (day-first or month-first, both ambiguous without
# locale info -- left to dateparser's own heuristics at parse time).
_DATE_TIME_PATTERN = re.compile(
    r"""
    (?P<date>
        \d{4}[/-]\d{1,2}[/-]\d{1,2}     # YYYY-MM-DD or YYYY/MM/DD
        |
        \d{1,2}[/-]\d{1,2}[/-]\d{4}     # MM-DD-YYYY / DD-MM-YYYY / MM/DD/YYYY
    )
    [\sT]+
    (?P<time>
        \d{1,2}:\d{2}(:\d{2})?         # HH:MM or HH:MM:SS
        \s*(AM|PM|am|pm)?
    )
    """,
    re.VERBOSE,
)


def _to_utc_iso(dt: datetime, assumed_tz: str = TIMESTAMP_OCR_ASSUMED_TZ) -> str:
    """Attaches `assumed_tz` if `dt` is naive (an OCR'd overlay or a plain
    caller-supplied string has no timezone marker of its own), then
    converts to UTC. A `dt` that already carries a timezone (e.g. a
    caller-supplied ISO 8601 string with an offset) is trusted as-is and
    just converted, not reinterpreted."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(assumed_tz))
    return dt.astimezone(timezone.utc).isoformat()


def extract_recorded_at(frame: Image.Image) -> datetime | None:
    """OCRs `frame` and returns the first date+time match found, parsed
    into a datetime -- or None if nothing matches or the match can't be
    parsed. Returns None rather than a best guess on failure: a wrong
    silently-stored timestamp would make a time-range filter confidently
    return the wrong clips, which is worse than the filter having no value
    to work with for that clip.
    """
    # Convert to grayscale before OCR -- found via testing that a full-color
    # frame (e.g. a saturated background behind a translucent overlay box)
    # can make Tesseract's binarization fail to find any text at all, where
    # the same frame in grayscale reads perfectly. Scoped to this function,
    # not app.pipeline.ocr's shared extract_text(), so general on-screen-text
    # search (tuned for slide/whiteboard content) is unaffected.
    text = extract_text([frame.convert("L")])
    match = _DATE_TIME_PATTERN.search(text)
    if not match:
        return None
    # STRICT_PARSING requires a complete day+month+year (rejects a partial
    # or ambiguous match instead of dateparser silently filling in
    # "today"'s date/year, which would be actively misleading here).
    return dateparser.parse(match.group(0), settings={"STRICT_PARSING": True})


def resolve_recorded_at(attributes: dict | None, first_frame: Image.Image | None, *, enabled: bool) -> str | None:
    """The one entry point app.pipeline.indexer calls: returns a UTC ISO
    8601 string for the clip's `recorded_at` payload field, or None if it
    can't be determined.

    Precedence: a caller-supplied `attributes.recorded_at` always wins over
    OCR and skips the OCR call entirely (explicit metadata over a
    best-effort guess, and cheaper). OCR only runs when `enabled`
    (ENABLE_TIMESTAMP_EXTRACTION) and no manual value was supplied.
    """
    manual = (attributes or {}).get("recorded_at")
    if manual:
        parsed = dateparser.parse(str(manual))
        return _to_utc_iso(parsed) if parsed else None
    if not enabled or first_frame is None:
        return None
    parsed = extract_recorded_at(first_frame)
    return _to_utc_iso(parsed) if parsed else None
