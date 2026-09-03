"""Speech-to-text for clips, gated by voice-activity detection.

Uses faster-whisper's built-in VAD filter (a bundled Silero VAD running on
onnxruntime, not torch -- no CUDA-dependency risk) so clips with no
detected speech never reach the Whisper model at all. Most CCTV/library
footage has no speech content worth transcribing; see vault note "Research"
section (day one) and the "Feasibility study" entry (2026-09-03) for why
this is a separate, optional signal from the CLIP visual embedding.
"""

from pathlib import Path

from faster_whisper import WhisperModel

from app.config import MIN_TRANSCRIPT_CHARS, WHISPER_MODEL_SIZE

_model = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_clip(clip_path: Path) -> str | None:
    """Returns the clip's transcribed speech, or None if no real speech was
    detected (VAD found nothing, or the transcript is too short to be
    meaningful rather than noise/hallucination)."""
    model = _load_model()
    segments, _ = model.transcribe(str(clip_path), vad_filter=True, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return text if len(text) >= MIN_TRANSCRIPT_CHARS else None
