from __future__ import annotations

import hashlib
import io
import json
import os
import random
import zipfile
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .config import DataConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_vis(seq: str) -> bool:
    return "_VIS" in seq


def _jpeg_size(data: bytes) -> Tuple[int, int]:
    """Parse JPEG SOF marker to return (width, height) without full decode."""
    i, n = 0, len(data)
    while i < n - 3:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9):          # SOI / EOI — no length field
            continue
        if i + 1 >= n:
            break
        length = (data[i] << 8) | data[i + 1]
        # All SOF variants that carry image dimensions
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 3] << 8) | data[i + 4]
            w = (data[i + 5] << 8) | data[i + 6]
            return w, h
        i += length
    raise ValueError("Could not parse JPEG dimensions")


def _any_overlap(
    x1: int, y1: int, x2: int, y2: int,
    bboxes_px: List[Tuple[float, float, float, float]],
) -> bool:
    """Return True if the crop [x1,y1,x2,y2] overlaps any bbox (cx,cy,bw,bh in px)."""
    for cx, cy, bw, bh in bboxes_px:
        bx1, by1 = cx - bw / 2, cy - bh / 2
        bx2, by2 = cx + bw / 2, cy + bh / 2
        if min(x2, bx2) > max(x1, bx1) and min(y2, by2) > max(y1, by1):
            return True
    return False


class _NullContext:
    """No-op context manager used when operating in directory mode."""
    def __enter__(self):
        return None
    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index(
    zip_path: Optional[str],
    data_dir: Optional[str],
    sequences: List[str],
    image_width: int,
    crop_size: int,
    pos_jitter: float,
    neg_per_pos: float,
    seed: int,
) -> List[Tuple]:
    """
    Build the static crop index: list of (img_relpath, x1, y1, x2, y2, label).

    Coordinates are in the space of images resized to image_width so that
    object scale is consistent across the different drone/resolution combos.
    """
    rng = random.Random(seed)
    crops: List[Tuple] = []
    half = crop_size // 2

    def open_zip():
        return zipfile.ZipFile(zip_path, "r") if zip_path else None

    with (open_zip() or _NullContext()) as zf:
        all_names = set(zf.namelist()) if zf is not None else None

        for seq in sequences:
            if zf is not None:
                jpgs = sorted(
                    n for n in all_names
                    if n.startswith(seq + "/") and n.endswith(".jpg")
                )
            else:
                seq_dir = os.path.join(data_dir, seq)
                jpgs = sorted(
                    os.path.join(seq, f)
                    for f in os.listdir(seq_dir)
                    if f.endswith(".jpg")
                )

            for jpg in jpgs:
                txt = jpg[:-4] + ".txt"

                # Fast JPEG dimension read (64 KB covers any SOF marker)
                if zf is not None:
                    with zf.open(jpg) as f:
                        header = f.read(65536)
                    try:
                        orig_w, orig_h = _jpeg_size(header)
                    except ValueError:
                        continue
                else:
                    with Image.open(os.path.join(data_dir, jpg)) as img:
                        orig_w, orig_h = img.size

                # Scale to target width, maintain aspect ratio
                scale = image_width / orig_w
                W = image_width
                H = int(orig_h * scale)

                # Load annotations: YOLO format (class cx cy w h), all normalized
                bboxes: List[Tuple[float, float, float, float]] = []
                txt_exists = (
                    (txt in all_names) if zf is not None
                    else os.path.isfile(os.path.join(data_dir, txt))
                )
                if txt_exists:
                    if zf is not None:
                        raw = zf.read(txt).decode("utf-8").strip()
                    else:
                        with open(os.path.join(data_dir, txt)) as fh:
                            raw = fh.read().strip()
                    for line in raw.splitlines():
                        parts = line.split()
                        if len(parts) == 5:
                            bboxes.append(tuple(map(float, parts[1:])))

                # Rescale bbox centers/dims to resized-image pixel coords
                bboxes_px: List[Tuple[float, float, float, float]] = [
                    (cx * W, cy * H, bw * W, bh * H) for cx, cy, bw, bh in bboxes
                ]

                max_jitter = int(pos_jitter * crop_size)

                # --- Positive crops (one per annotation, with random jitter) ---
                for cx_px, cy_px, _, _ in bboxes_px:
                    dx = rng.randint(-max_jitter, max_jitter)
                    dy = rng.randint(-max_jitter, max_jitter)
                    # Clamp so the full crop stays inside the resized image
                    cx_c = int(max(half, min(W - half, cx_px + dx)))
                    cy_c = int(max(half, min(H - half, cy_px + dy)))
                    x1, y1 = cx_c - half, cy_c - half
                    crops.append((jpg, x1, y1, x1 + crop_size, y1 + crop_size, 1))

                # --- Negative crops (random, zero overlap with any annotation) ---
                if W <= crop_size or H <= crop_size:
                    continue
                n_neg = max(1, int(len(bboxes_px) * neg_per_pos + 0.5)) if bboxes_px else 2
                added = 0
                for _ in range(n_neg * 30):
                    if added >= n_neg:
                        break
                    cx_r = rng.randint(half, W - half)
                    cy_r = rng.randint(half, H - half)
                    x1, y1 = cx_r - half, cy_r - half
                    if not _any_overlap(x1, y1, x1 + crop_size, y1 + crop_size, bboxes_px):
                        crops.append((jpg, x1, y1, x1 + crop_size, y1 + crop_size, 0))
                        added += 1

    return crops


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WiSARDClassDataset(Dataset):
    """
    Binary crop classification dataset built from WiSARD VIS sequences.

    Each sample is a (crop_size × crop_size) patch from a VIS image, labelled
    1 (person present) or 0 (background).  Full images are first resized to
    image_width so that person scale is consistent across drone/resolution
    combinations.  The crop index is built once and cached to disk.
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str,               # "train" or "val"
        cache_dir: str = ".cache",
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.cfg = cfg
        self.split = split

        # Enumerate VIS sequences; split deterministically by sorted name
        all_seqs = sorted(self._list_vis_sequences())
        n_val = max(1, int(len(all_seqs) * cfg.val_fraction))
        val_seqs   = all_seqs[-n_val:]
        train_seqs = all_seqs[:-n_val]
        self.sequences = val_seqs if split == "val" else train_seqs

        self.crops = self._load_or_build_index(cache_dir)

        normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if split == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transform = T.Compose([T.ToTensor(), normalize])

        # ZipFile handle: opened lazily so each DataLoader worker gets its own handle
        self._zf: Optional[zipfile.ZipFile] = None

    # ------------------------------------------------------------------

    def _list_vis_sequences(self) -> List[str]:
        if self.cfg.data_dir:
            return [
                d for d in os.listdir(self.cfg.data_dir)
                if os.path.isdir(os.path.join(self.cfg.data_dir, d)) and _is_vis(d)
            ]
        with zipfile.ZipFile(self.cfg.zip_path, "r") as zf:
            return sorted({
                n.split("/")[0]
                for n in zf.namelist()
                if "/" in n and _is_vis(n.split("/")[0])
            })

    def _cache_key(self) -> str:
        cfg = self.cfg
        raw = (
            f"{cfg.zip_path}|{cfg.data_dir}|{cfg.image_width}|{cfg.crop_size}"
            f"|{cfg.pos_jitter}|{cfg.neg_per_pos}|{cfg.seed}"
            f"|{self.split}|{'|'.join(self.sequences)}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _load_or_build_index(self, cache_dir: str) -> List[Tuple]:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"wisard_index_{self.split}_{self._cache_key()}.json")
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                return [tuple(row) for row in json.load(f)]

        print(f"Building {self.split} index over {len(self.sequences)} sequences...")
        crops = _build_index(
            zip_path=self.cfg.zip_path if not self.cfg.data_dir else None,
            data_dir=self.cfg.data_dir,
            sequences=self.sequences,
            image_width=self.cfg.image_width,
            crop_size=self.cfg.crop_size,
            pos_jitter=self.cfg.pos_jitter,
            neg_per_pos=self.cfg.neg_per_pos,
            seed=self.cfg.seed,
        )
        with open(cache_path, "w") as f:
            json.dump([list(c) for c in crops], f)
        n_pos = sum(1 for c in crops if c[5] == 1)
        print(f"  -> {len(crops)} crops  ({n_pos} pos / {len(crops) - n_pos} neg)")
        return crops

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, x1, y1, x2, y2, label = self.crops[idx]

        # Lazy-open ZIP once per DataLoader worker (each worker forks with _zf=None)
        if self.cfg.data_dir is None and self._zf is None:
            self._zf = zipfile.ZipFile(self.cfg.zip_path, "r")

        # Load image
        if self._zf is not None:
            data = self._zf.read(img_path)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.open(os.path.join(self.cfg.data_dir, img_path)).convert("RGB")

        # Resize to standard width (same scale used during index building)
        orig_w, orig_h = img.size
        new_h = int(orig_h * self.cfg.image_width / orig_w)
        img = img.resize((self.cfg.image_width, new_h), Image.BILINEAR)

        crop = img.crop((x1, y1, x2, y2))
        # Guard against off-by-one from integer rounding
        if crop.size != (self.cfg.crop_size, self.cfg.crop_size):
            crop = crop.resize((self.cfg.crop_size, self.cfg.crop_size), Image.BILINEAR)

        return self.transform(crop), label
