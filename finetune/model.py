from __future__ import annotations

import torch
import torch.nn as nn

from swin2_utils import MySwinV2, SwinV2Cfg

from .config import ModelConfig


def build_model(cfg: ModelConfig, device: str = "cpu") -> MySwinV2:
    """
    Instantiate MySwinV2 from a pre-converted checkpoint, replace the
    1000-class head with a single-logit binary head, and optionally freeze
    the backbone.

    The returned model outputs raw logits of shape (B,) — pair with
    BCEWithLogitsLoss.

    The checkpoint at cfg.pretrained_path should be the MySwinV2 state dict
    saved after running the HF weight conversion (loading.ipynb):

        torch.save(model.state_dict(), "swinv2_wisard_pretrained.pt")
    """
    # Reconstruct the same config used when the checkpoint was saved.
    # Tiny variant defaults match microsoft/swinv2-tiny-patch4-window16-256.
    swin_cfg = SwinV2Cfg(hidden_dropout_prob=cfg.dropout)

    model = MySwinV2(swin_cfg, num_classes=1000)
    state = torch.load(cfg.pretrained_path, map_location="cpu")
    model.load_state_dict(state)

    # Replace classification head with a single-logit binary head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad_(False)

    return model.to(device)
