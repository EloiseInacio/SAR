"""
One-time script: download microsoft/swinv2-tiny-patch4-window16-256, convert
weights to MySwinV2 format, and save the state dict to disk.

Run once before finetuning:
    python save_pretrained.py
"""
import torch
from transformers import Swinv2ForImageClassification

from hf_utils import convert_hf_swinv2_state_dict
from swin2_utils import MySwinV2, SwinV2Cfg

HF_MODEL_ID = "microsoft/swinv2-tiny-patch4-window16-256"
OUT_PATH    = "swinv2_wisard_pretrained.pt"

hf_model = Swinv2ForImageClassification.from_pretrained(HF_MODEL_ID)
hf_cfg   = hf_model.config

swin_cfg = SwinV2Cfg(
    image_size=hf_cfg.image_size,
    patch_size=hf_cfg.patch_size,
    num_channels=hf_cfg.num_channels,
    embed_dim=hf_cfg.embed_dim,
    depths=tuple(hf_cfg.depths),
    num_heads=tuple(hf_cfg.num_heads),
    window_size=hf_cfg.window_size,
    pretrained_window_sizes=tuple(hf_cfg.pretrained_window_sizes),
    mlp_ratio=hf_cfg.mlp_ratio,
    qkv_bias=hf_cfg.qkv_bias,
    drop_path_rate=hf_cfg.drop_path_rate,
    layer_norm_eps=hf_cfg.layer_norm_eps,
)

model      = MySwinV2(swin_cfg, num_classes=1000)
state_dict = convert_hf_swinv2_state_dict(hf_model.state_dict(), model)
model.load_state_dict(state_dict)

torch.save(model.state_dict(), OUT_PATH)
print(f"Saved {OUT_PATH}")
