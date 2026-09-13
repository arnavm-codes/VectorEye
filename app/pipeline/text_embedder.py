"""Sentence embeddings for general (non-CLIP) text -- transcript and OCR'd
on-screen text both live here, sharing one model and one embedding
function, but in separate named Qdrant vectors ("transcript" / "ocr_text")
since they're different signals even though the embedding model is the same.

CLIP's text encoder is tuned for image-caption alignment, not general
sentence semantics, so it can't be reused for either (confirmed in the
vault note's "Feasibility study" entry, 2026-09-03, for transcripts; the
same reasoning applies unchanged to OCR'd text). This module is the
transcript/OCR-side counterpart to app/pipeline/embedder.py's CLIP
image/text embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import TRANSCRIPT_EMBED_MODEL

_model = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(TRANSCRIPT_EMBED_MODEL, device="cpu")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embeds any general text (transcript, OCR'd text, or a search query
    for either) into the shared vector space. L2-normalized, matching the
    Cosine-distance collection."""
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)


# Kept as an alias -- existing callers (app.pipeline.indexer,
# app.api.search) use this name; embed_text() is the same function under
# the more accurate general-purpose name now that OCR text uses it too.
embed_transcript_text = embed_text
