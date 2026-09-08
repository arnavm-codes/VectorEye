"""Attribute verification via open-vocabulary object detection.

Localizes the query's object noun in a clip's sampled frames with
YOLO-World, then verifies the attribute with CLIP only inside that crop --
a single-concept comparison, so CLIP's cross-modal attribute-binding
weakness (which only manifests when multiple objects/attributes compete
inside one global embedding) doesn't come into play. See vault note
"Attribute-binding / bag-of-words retrieval failure" and the YOLO-World-L
model-choice comparison, both dated 2026-09-08.
"""

import numpy as np
from PIL import Image
from ultralytics import YOLOWorld

from app.config import YOLO_WORLD_MODEL
from app.pipeline.embedder import embed_image, embed_text

_detector = None
_detector_classes: list[str] | None = None


def _load_detector():
    global _detector
    if _detector is None:
        _detector = YOLOWorld(YOLO_WORLD_MODEL)
    return _detector


def _set_detector_classes(obj: str) -> None:
    # set_classes() re-embeds the text prompt, so skip the call (and its
    # cost) if the class is already the one loaded from the previous frame.
    global _detector_classes
    if _detector_classes != [obj]:
        _load_detector().set_classes([obj])
        _detector_classes = [obj]


def _score_attribute(crop: Image.Image, attribute: str, obj: str) -> float:
    """CLIP cosine similarity between `crop` and "a photo of a {attribute}
    {obj}" -- reuses the same CLIP/Long-CLIP backend already loaded for the
    main search path (app.pipeline.embedder), so this doesn't introduce a
    second embedding space to keep in sync.
    """
    image_embed = embed_image(crop)
    text_embed = embed_text(f"a photo of a {attribute} {obj}")
    return float(np.dot(image_embed, text_embed))


def verify_attribute_in_frames(frames: list[Image.Image], attribute: str, obj: str) -> float:
    """Returns the best attribute-match score across every detected
    instance of `obj` across `frames`.

    Falls back to a whole-frame CLIP score (the same global-embedding
    comparison this feature exists to improve on) if `obj` was never
    detected in any frame -- NOT because the crop-verify approach failed
    for this clip specifically, but because it's indistinguishable from a
    real, confirmed failure mode: YOLO-World's zero-shot vocabulary can
    silently return zero detections for an exact noun phrasing it wasn't
    trained on (e.g. "tshirt"/"t-shirt" score 0 detections even on an image
    with two clearly visible shirts, while "shirt" alone detects both --
    verified directly, see vault note 2026-09-08). Hard-returning 0.0 in
    that case would silently zero out an otherwise-correct clip rather than
    "no worse than before this feature existed" -- see vault note's bug
    writeup for the full repro.
    """
    _set_detector_classes(obj)
    detector = _load_detector()

    best_score = 0.0
    object_detected = False
    for frame in frames:
        frame_np = np.array(frame)
        results = detector.predict(frame_np, verbose=False)[0]
        for box in results.boxes:
            object_detected = True
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            crop = frame.crop((x1, y1, x2, y2))
            if crop.width < 2 or crop.height < 2:
                continue
            score = _score_attribute(crop, attribute, obj)
            best_score = max(best_score, score)

    if not object_detected:
        return max(_score_attribute(frame, attribute, obj) for frame in frames)
    return best_score
