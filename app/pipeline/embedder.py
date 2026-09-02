"""CLIP-based embedding: sparse frame sampling + mean-pool per clip.

No captioning, no LLM -- a clip's embedding is the mean of a few sampled
frame embeddings, living in the same joint space as CLIP's text encoder.
See vault note "Research" section for why this is sufficient without an
explicit transcription/captioning step.
"""

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED, FRAMES_PER_CLIP

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None
_tokenizer = None


def _load_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=_device
        )
        _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        _model.eval()
    return _model, _preprocess, _tokenizer


def sample_frames(clip_path, n_frames: int = FRAMES_PER_CLIP) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def embed_clip(clip_path) -> np.ndarray:
    """Returns a single mean-pooled, L2-normalized embedding vector for a clip."""
    model, preprocess, _ = _load_model()
    frames = sample_frames(clip_path)
    if not frames:
        raise ValueError(f"No frames could be read from {clip_path}")

    batch = torch.stack([preprocess(f) for f in frames]).to(_device)
    with torch.no_grad():
        frame_embeds = model.encode_image(batch)
        frame_embeds = frame_embeds / frame_embeds.norm(dim=-1, keepdim=True)
        clip_embed = frame_embeds.mean(dim=0)
        clip_embed = clip_embed / clip_embed.norm()

    return clip_embed.cpu().numpy()


def embed_text(query: str) -> np.ndarray:
    """Embeds a text query into the same CLIP space as embed_clip()."""
    model, _, tokenizer = _load_model()
    tokens = tokenizer([query]).to(_device)
    with torch.no_grad():
        text_embed = model.encode_text(tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    return text_embed[0].cpu().numpy()
