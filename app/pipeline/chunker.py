"""Splits raw video files into fixed-length clips via ffmpeg.

Fixed 10s chunking, no overlap and no scene detection -- deliberate POC
simplification (see vault note "Chunking" section for the accepted
boundary-splitting tradeoff).
"""

import subprocess
from pathlib import Path

from app.config import CLIP_DURATION_SECONDS, CLIPS_DIR, RAW_VIDEOS_DIR


def chunk_video(video_path: Path, out_dir: Path, clip_seconds: int = CLIP_DURATION_SECONDS) -> list[Path]:
    """Chunk one video into fixed-length clips, named `<stem>_clip000.mp4`, etc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    pattern = out_dir / f"{stem}_clip%03d.mp4"

    # Re-encode (not -c copy) with keyframes forced at each split point --
    # stream-copy segmenting only cuts at existing keyframes, which silently
    # produces uneven clip lengths on sources with long GOPs (common on
    # CCTV/NVR encoders). Forcing keyframes guarantees exact clip_seconds
    # boundaries, which the indexer's start_ts/end_ts metadata assumes.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-map", "0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-force_key_frames", f"expr:gte(t,n_forced*{clip_seconds})",
            "-c:a", "aac",
            "-f", "segment",
            "-segment_time", str(clip_seconds),
            # Without this, floating-point rounding on the forced keyframe's
            # pts can make the segment muxer miss a boundary that lands
            # exactly on clip_seconds, silently merging it into the next
            # segment (reproduced: a keyframe at exactly t=10.000000 was
            # skipped, producing a 20s clip instead of two 10s ones).
            "-segment_time_delta", "0.5",
            "-reset_timestamps", "1",
            "-segment_format_options", "movflags=+faststart",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob(f"{stem}_clip*.mp4"))


def chunk_all(raw_dir: Path = RAW_VIDEOS_DIR, out_dir: Path = CLIPS_DIR) -> list[Path]:
    video_paths = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
    )
    all_clips: list[Path] = []
    for video_path in video_paths:
        print(f"Chunking {video_path.name} ...")
        clips = chunk_video(video_path, out_dir)
        print(f"  -> {len(clips)} clips")
        all_clips.extend(clips)
    return all_clips


if __name__ == "__main__":
    chunk_all()
