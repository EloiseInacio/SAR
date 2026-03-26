from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    zip_path: str = "WiSARDv1.zip"
    data_dir: Optional[str] = None       # pre-extracted dir; faster than ZIP streaming
    image_width: int = 2048              # resize full images to this width before cropping
    crop_size: int = 256                 # square crop fed to MySwinV2 (must match model image_size)
    pos_jitter: float = 0.3             # max center jitter as fraction of crop_size
    neg_per_pos: float = 2.0            # negative crops generated per positive crop
    val_fraction: float = 0.2           # fraction of VIS sequences held out for val (by sequence)
    seed: int = 42
    num_workers: int = 4                 # set 0 when reading directly from ZIP


@dataclass
class ModelConfig:
    pretrained_path: str = "swinv2_wisard_pretrained.pt"  # MySwinV2 state dict (pre-converted from HF)
    freeze_backbone: bool = False        # freeze all layers except the replaced head
    dropout: float = 0.0                 # hidden_dropout_prob passed to SwinV2Cfg


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-5                     # head learning rate
    backbone_lr_multiplier: float = 0.1  # backbone LR = lr * multiplier
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    pos_weight: float = 2.0              # BCEWithLogitsLoss positive class weight
    clip_grad: float = 5.0
    checkpoint_dir: str = "checkpoints/wisard"
    log_every: int = 100                 # training steps between console log lines
    device: str = "auto"                 # "auto" -> cuda if available, else cpu


@dataclass
class FinetuneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
