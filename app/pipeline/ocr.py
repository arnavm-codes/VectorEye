"""On-screen text extraction (OCR) -- a third independent search signal
alongside visual (CLIP) and speech-content (Whisper transcript), following
the same overall shape app.pipeline.transcriber established: run over a
sparse sample of a clip's frames, produce text, embed it separately, fuse
at query time via RRF. Uses its own sparser frame sample
(OCR_FRAMES_PER_CLIP, see app.pipeline.indexer) rather than reusing the
denser one taken for the visual embedding -- on-screen text is typically
static across many consecutive frames, so fewer frames are enough to catch
it, same reasoning as attribute verification's sparser sampling. See vault
note "OCR effort kicked off" entry, 2026-09-13, for the engine comparison
behind the default.

Backend dispatch follows this project's house convention (same entry):
one config value (OCR_BACKEND) picks an engine *family*, one
`_load_<backend>()` function per family, one dispatch point
(`_load_backend()`), and a single public interface (`extract_text()`) that
every caller uses without knowing which engine is active. Adding a second
engine (e.g. "paddleocr") means adding one `_load_paddleocr()` function and
one branch here -- no changes anywhere else in the codebase. An
unrecognized OCR_BACKEND fails loudly (ValueError), not a silent fallback.
"""

from PIL import Image

from app.config import OCR_BACKEND

_backend = None


def _load_tesseract():
    import pytesseract

    # Fails loudly and early (at first use, not buried inside a per-frame
    # try/except) if the tesseract-ocr system binary isn't installed --
    # pytesseract is only a thin wrapper around that binary, it doesn't
    # bundle it. See README's OCR setup note.
    pytesseract.get_tesseract_version()

    def run(image: Image.Image) -> str:
        return pytesseract.image_to_string(image)

    return run


def _load_backend():
    global _backend
    if _backend is None:
        if OCR_BACKEND == "tesseract":
            _backend = _load_tesseract()
        else:
            raise ValueError(f"Unrecognized OCR_BACKEND {OCR_BACKEND!r} (expected 'tesseract')")
    return _backend


def extract_text(frames: list[Image.Image]) -> str:
    """Runs OCR over `frames` (typically the same sparse frame sample used
    elsewhere for a clip) and returns the deduplicated on-screen text found
    across all of them, newline-joined.

    Dedup matters because on-screen text is usually static across many
    consecutive frames (a slide held for several seconds) -- without it,
    the same line would repeat once per frame in the stored/embedded text,
    which both wastes embedding signal on repetition and would bias the
    text toward whichever slide happened to get sampled more densely.
    Order-preserving (first-seen order), not sorted, so a slide sequence
    still reads in the order it was shown.
    """
    run = _load_backend()
    seen: dict[str, None] = {}
    for frame in frames:
        text = run(frame).strip()
        for line in text.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen[line] = None
    return "\n".join(seen)
