"""Sliding-window inference over full UAV images."""
from __future__ import annotations

from typing import List, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

from .dataset import IMAGENET_MEAN, IMAGENET_STD


_PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Type alias: detection = (x1, y1, x2, y2, confidence) in resized-image pixels
Detection = Tuple[int, int, int, int, float]


def sliding_window_detect(
    model: nn.Module,
    image: Image.Image,
    image_width: int = 2048,
    crop_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    batch_size: int = 64,
    device: str = "cpu",
) -> List[Detection]:
    """
    Classify every stride-spaced crop of `image` and return NMS-filtered
    detections.

    The image is first resized to `image_width` (same preprocessing used
    during training).  Returned coordinates are in that resized space; to
    map back to the original image multiply by orig_w / image_width.
    """
    orig_w, orig_h = image.size
    new_h = int(orig_h * image_width / orig_w)
    image = image.resize((image_width, new_h), Image.BILINEAR)
    W, H = image.size

    # All top-left corner positions that keep the crop fully inside the image
    xs = list(range(0, W - crop_size + 1, stride))
    ys = list(range(0, H - crop_size + 1, stride))
    windows: List[Tuple[int, int]] = [(x, y) for y in ys for x in xs]

    model.eval()
    detections: List[Detection] = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_coords = windows[i: i + batch_size]
            crops = torch.stack([
                _PREPROCESS(image.crop((x, y, x + crop_size, y + crop_size)))
                for x, y in batch_coords
            ])
            logits = model(crops.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().tolist()
            for (x, y), prob in zip(batch_coords, probs):
                if prob >= threshold:
                    detections.append((x, y, x + crop_size, y + crop_size, prob))

    return nms(detections, iou_threshold=0.3)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
    """Greedy NMS sorted by descending confidence."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept: List[Detection] = []
    suppressed = [False] * len(detections)
    for i, d in enumerate(detections):
        if suppressed[i]:
            continue
        kept.append(d)
        for j in range(i + 1, len(detections)):
            if not suppressed[j] and _iou(d, detections[j]) >= iou_threshold:
                suppressed[j] = True
    return kept


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
