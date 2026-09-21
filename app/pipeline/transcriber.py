"""Speech-to-text for clips, gated by voice-activity detection.

Uses faster-whisper's built-in VAD filter (a bundled Silero VAD running on
onnxruntime, not torch -- no CUDA-dependency risk) so clips with no
detected speech never reach the Whisper model at all. Most CCTV/library
footage has no speech content worth transcribing; see vault note "Research"
section (day one) and the "Feasibility study" entry (2026-09-03) for why
this is a separate, optional signal from the CLIP visual embedding.
"""

from pathlib import Path

import av  # already installed as a faster-whisper dependency
from faster_whisper import WhisperModel

from app.config import MIN_TRANSCRIPT_CHARS, WHISPER_MODEL_SIZE

_model = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def _has_audio_stream(clip_path: Path) -> bool:
    """True if the file contains at least one audio stream. Video-only clips
    (common for CCTV/NVR footage) have none, and faster-whisper's decoder
    crashes on them with `IndexError: tuple index out of range` instead of
    returning an empty result -- which would abort the whole indexing run."""
    with av.open(str(clip_path)) as container:
        return len(container.streams.audio) > 0


def transcribe_clip(clip_path: Path) -> str | None:
    """Returns the clip's transcribed speech, or None if there is nothing to
    transcribe: the clip has no audio stream at all, no real speech was
    detected (VAD found nothing), or the transcript is too short to be
    meaningful rather than noise/hallucination."""
    if not _has_audio_stream(clip_path):
        return None
    model = _load_model()
    segments, _ = model.transcribe(str(clip_path), vad_filter=True, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return text if len(text) >= MIN_TRANSCRIPT_CHARS else None
