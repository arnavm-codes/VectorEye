"""Sentence embeddings for transcript text -- a separate model and vector
space from CLIP's visual embeddings.

CLIP's text encoder is tuned for image-caption alignment, not general
sentence semantics, so it can't be reused to embed transcripts (confirmed
in the vault note's "Feasibility study" entry, 2026-09-03). This module is
the transcript-side counterpart to app/pipeline/embedder.py's CLIP
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


def embed_transcript_text(text: str) -> np.ndarray:
    """Embeds transcript text or a search query into the shared transcript
    vector space. L2-normalized, matching the Cosine-distance collection."""
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)
