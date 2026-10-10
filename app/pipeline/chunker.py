"""Splits raw video files into fixed-length clips via ffmpeg.

Two overlapping chunking passes per video -- start times 0/10/20s and
CHUNK_OVERLAP_SECONDS/+10/+20s -- so an event spanning a fixed-chunk boundary
still lands cleanly inside at least one clip instead of being split across
two weak-scoring ones (see vault note "Chunking" section). No scene
detection -- deliberate POC simplification, accepted tradeoff.

Clips are named `<stem>_clip<start_ts>.mp4` where <start_ts> is the clip's
actual start second in the source video (zero-padded), so the indexer can
read start_ts/end_ts straight from the filename regardless of which pass
produced it.
"""

import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from app import storage
from app.config import CHUNK_OVERLAP_SECONDS, CLIP_DURATION_SECONDS, RAW_VIDEOS_BUCKET

_START_TS_DIGITS = 6  # supports source videos up to ~11.5 days


def _run_segment_pass(video_path: Path, out_dir: Path, clip_seconds: int, offset_seconds: int) -> list[Path]:
    """Runs one ffmpeg segment pass starting at offset_seconds, returns final
    (renamed) clip paths sorted by start time."""
    stem = video_path.stem
    tmp_prefix = f"{stem}_tmp{offset_seconds}"
    tmp_pattern = out_dir / f"{tmp_prefix}_%03d.mp4"

    cmd = ["ffmpeg", "-y"]
    if offset_seconds:
        # Output seeking (-ss after -i), not input seeking: this pass
        # re-encodes anyway, so paying the decode cost for a frame-accurate
        # start point is worth it over the faster but keyframe-snapped
        # input-seek behavior.
        cmd += ["-i", str(video_path), "-ss", str(offset_seconds)]
    else:
        cmd += ["-i", str(video_path)]

    # Re-encode (not -c copy) with keyframes forced at each split point --
    # stream-copy segmenting only cuts at existing keyframes, which silently
    # produces uneven clip lengths on sources with long GOPs (common on
    # CCTV/NVR encoders). Forcing keyframes guarantees exact clip_seconds
    # boundaries, which the indexer's start_ts/end_ts metadata assumes.
    cmd += [
        "-map", "0",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-force_key_frames", f"expr:gte(t,n_forced*{clip_seconds})",
        "-c:a", "aac",
        "-f", "segment",
        "-segment_time", str(clip_seconds),
        # Without this, floating-point rounding on the forced keyframe's pts
        # can make the segment muxer miss a boundary that lands exactly on
        # clip_seconds, silently merging it into the next segment
        # (reproduced: a keyframe at exactly t=10.000000 was skipped,
        # producing a 20s clip instead of two 10s ones).
        "-segment_time_delta", "0.5",
        "-reset_timestamps", "1",
        "-segment_format_options", "movflags=+faststart",
        str(tmp_pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    tmp_clips = sorted(out_dir.glob(f"{tmp_prefix}_*.mp4"))
    final_clips = []
    for seq, tmp_path in enumerate(tmp_clips):
        start_ts = offset_seconds + seq * clip_seconds
        final_path = out_dir / f"{stem}_clip{start_ts:0{_START_TS_DIGITS}d}.mp4"
        tmp_path.rename(final_path)
        final_clips.append(final_path)
    return final_clips


def chunk_video(
    video_path: Path,
    out_dir: Path,
    clip_seconds: int = CLIP_DURATION_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
) -> list[Path]:
    """Chunk one video into two overlapping sets of fixed-length clips
    (start times 0/clip_seconds/... and overlap_seconds/+clip_seconds/...),
    named `<stem>_clip<start_ts>.mp4`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = _run_segment_pass(video_path, out_dir, clip_seconds, offset_seconds=0)
    if overlap_seconds:
        clips += _run_segment_pass(video_path, out_dir, clip_seconds, offset_seconds=overlap_seconds)
    return sorted(clips)


def chunk_video_key(video_key: str) -> list[str]:
    """Chunk one raw video (an object key in the raw-videos bucket) into clips
    uploaded to the clips bucket. Returns the uploaded clip keys. Temp files
    are always cleaned up."""
    print(f"Chunking s3://{RAW_VIDEOS_BUCKET}/{video_key} ...")
    uploaded: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vectoreye_chunk_") as tmp:
        tmp_dir = Path(tmp)
        local_video = storage.download(RAW_VIDEOS_BUCKET, video_key, tmp_dir / Path(video_key).name)
        clips = chunk_video(local_video, tmp_dir / "clips")
        # Clips keep the raw video's own prefix (cam1/clip.mp4 -> cam1/clip_clip000000.mp4).
        # Keyed by file name alone, two videos with the same name under different
        # prefixes (e.g. one 00001.mp4 per camera) would overwrite each other's clips.
        prefix = PurePosixPath(video_key).parent
        for clip in clips:
            key = clip.name if str(prefix) == "." else f"{prefix.as_posix()}/{clip.name}"
            uploaded.append(storage.upload_clip(clip, key))
    print(f"  -> {len(uploaded)} clips uploaded")
    return uploaded


def chunk_all(video_key: str | None = None) -> list[str]:
    """Chunk raw videos from the raw-videos bucket into clips uploaded to the
    clips bucket. Processes `video_key` only if given, otherwise every video.
    Returns the uploaded clip keys."""
    video_keys = [video_key] if video_key else storage.list_raw_videos()
    uploaded: list[str] = []
    for key in video_keys:
        uploaded.extend(chunk_video_key(key))
    return uploaded


if __name__ == "__main__":
    chunk_all()
