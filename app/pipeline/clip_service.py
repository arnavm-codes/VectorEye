"""On-demand clips: cut a window out of a raw video with ffmpeg, cache it on local disk.

Replaces the pre-cut clips bucket. A clip is identified by (video_key, start, end) in
whole seconds -- exactly what each Qdrant point already stores.

* Cuts are re-encoded (H.264/AAC, faststart) rather than stream-copied: copy snaps to
  keyframes, which on long-GOP CCTV footage can be seconds off, and re-encoding also
  makes HEVC / .mkv / .avi sources playable in a browser.
* ffmpeg reads the raw video over an internal presigned URL, so only the needed byte
  ranges are fetched -- it never downloads the whole recording.
* Results are cached under CLIP_CACHE_DIR, keyed by (video_key, etag, start, end) so a
  replaced video can never serve a stale clip, and trimmed LRU to CLIP_CACHE_MAX_MB.
* URLs handed to clients are HMAC-signed and expiring, valid for one window only, so the
  endpoint can't be used to pull arbitrary parts of a recording.
"""

import hashlib
import hmac
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

from botocore.exceptions import ClientError

from app import storage
from app.config import (
    API_PUBLIC_URL,
    CLIP_CACHE_DIR,
    CLIP_CACHE_MAX_MB,
    CLIP_CUT_CONCURRENCY,
    CLIP_CUT_TIMEOUT_SECONDS,
    CLIP_MAX_SECONDS,
    CLIP_SIGNING_SECRET,
    PRESIGN_EXPIRY_SECONDS,
)


_MIN_CLIP_BYTES = 2048  # ffmpeg exits 0 with a stub file when the window is past the end of the video
_cut_slots = threading.Semaphore(CLIP_CUT_CONCURRENCY)
_key_locks = [threading.Lock() for _ in range(64)]  # striped: same clip -> same lock, bounded memory
_evict_lock = threading.Lock()


class ClipConfigError(RuntimeError):
    """CLIP_SIGNING_SECRET is not set."""


def check_configured() -> None:
    """Fail fast (API startup) instead of on the first search."""
    _secret()


def _secret() -> bytes:
    if not CLIP_SIGNING_SECRET:
        raise ClipConfigError(
            "CLIP_SIGNING_SECRET is not set. Run setup.sh, or set it in .env (e.g. `openssl rand -hex 32`)."
        )
    return CLIP_SIGNING_SECRET.encode()


class ClipInvalid(ValueError):
    """Bad window (negative, empty, or longer than CLIP_MAX_SECONDS)."""


class ClipNotFound(LookupError):
    """Raw video doesn't exist, or the window lies outside it."""


class ClipError(RuntimeError):
    """ffmpeg failed or timed out."""


def _signature(video_key: str, start: int, end: int, exp: int) -> str:
    msg = f"{video_key}\n{start}\n{end}\n{exp}".encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def signed_clip_url(video_key: str, start: int, end: int, expires_in: int = PRESIGN_EXPIRY_SECONDS) -> str:
    exp = int(time.time()) + expires_in
    query = urlencode({"video_key": video_key, "start": start, "end": end, "exp": exp,
                       "sig": _signature(video_key, start, end, exp)})
    return f"{API_PUBLIC_URL}/clip?{query}"


def verify_signature(video_key: str, start: int, end: int, exp: int, sig: str) -> bool:
    return exp >= time.time() and hmac.compare_digest(_signature(video_key, start, end, exp), sig)


def _validate(start: int, end: int) -> None:
    if start < 0 or end <= start:
        raise ClipInvalid(f"invalid window {start}-{end}")
    if end - start > CLIP_MAX_SECONDS:
        raise ClipInvalid(f"window longer than {CLIP_MAX_SECONDS}s")


def _cache_path(video_key: str, etag: str, start: int, end: int) -> Path:
    digest = hashlib.sha256(f"{video_key}|{etag}|{start}|{end}".encode()).hexdigest()
    return CLIP_CACHE_DIR / f"{digest}.mp4"


def _cut(video_key: str, start: int, end: int, out: Path) -> None:
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        # -ss before -i: fast range-based seek; with re-encoding ffmpeg still decodes up to
        # the exact start frame, so the cut is frame-accurate even on long-GOP sources.
        "-ss", str(start), "-i", storage.raw_video_internal_url(video_key),
        "-t", str(end - start),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        "-f", "mp4", str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=CLIP_CUT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise ClipError(f"ffmpeg timed out after {CLIP_CUT_TIMEOUT_SECONDS}s") from exc
    if proc.returncode != 0:
        raise ClipError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-300:]}")


def _evict(keep: Path) -> None:
    """Trim the cache to CLIP_CACHE_MAX_MB, least recently used first."""
    with _evict_lock:
        files = [(f.stat().st_mtime, f.stat().st_size, f) for f in CLIP_CACHE_DIR.glob("*.mp4")]
        total = sum(size for _, size, _ in files)
        limit = CLIP_CACHE_MAX_MB * 1024 * 1024
        for _, size, f in sorted(files):
            if total <= limit:
                break
            if f != keep:
                f.unlink(missing_ok=True)
                total -= size


def get_clip_file(video_key: str, start: int, end: int) -> Path:
    """Local path of the clip, cutting it first if it isn't cached."""
    _validate(start, end)
    try:
        etag, _size = storage.raw_video_fingerprint(video_key)
    except ClientError as exc:
        raise ClipNotFound(f"raw video not found: {video_key}") from exc

    path = _cache_path(video_key, etag, start, end)
    if path.exists():
        os.utime(path)  # LRU touch
        return path

    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _key_locks[int(path.stem[:8], 16) % len(_key_locks)]:
        if path.exists():  # another request cut it while we waited
            return path
        with _cut_slots:
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.part")
            try:
                _cut(video_key, start, end, tmp)
                if not tmp.exists() or tmp.stat().st_size < _MIN_CLIP_BYTES:
                    raise ClipNotFound(f"window {start}-{end}s is outside {video_key}")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
    _evict(keep=path)
    return path
