"""Fine-tune MySwinV2 on WiSARD as a binary person/background classifier."""
from __future__ import annotations

import math
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from huggingface_hub import HfApi, create_repo, upload_file, login

from finetune.config import FinetuneConfig
from finetune.dataset import WiSARDClassDataset
from finetune.model import build_model


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _set_lr(optimizer: torch.optim.Optimizer, step: int,
            warmup_steps: int, total_steps: int, base_lrs: List[float]) -> None:
    """Linear warmup then cosine decay, applied in-place to all param groups."""
    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _make_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("head")
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("head")
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_multiplier},
            {"params": head_params,     "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    total = correct = tp = fp = fn = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs).squeeze(1)
        preds  = (logits > 0.0).long()

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        tp      += ((preds == 1) & (labels == 1)).sum().item()
        fp      += ((preds == 1) & (labels == 0)).sum().item()
        fn      += ((preds == 0) & (labels == 1)).sum().item()

        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)

    # ROC-AUC via trapezoidal rule over 200 thresholds
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels).float()
    n_pos = labels_cat.sum().item()
    n_neg = len(labels_cat) - n_pos
    thresholds = torch.linspace(logits_cat.min(), logits_cat.max(), 200)
    fprs, tprs = [], []
    for t in thresholds:
        p = (logits_cat > t).float()
        tprs.append(((p == 1) & (labels_cat == 1)).sum().item() / max(1, n_pos))
        fprs.append(((p == 1) & (labels_cat == 0)).sum().item() / max(1, n_neg))
    # Sort by FPR for correct trapezoidal integration
    roc = sorted(zip(fprs, tprs))
    auc = sum(
        0.5 * (roc[i][1] + roc[i + 1][1]) * abs(roc[i + 1][0] - roc[i][0])
        for i in range(len(roc) - 1)
    )

    return {
        "acc":       correct / max(1, total),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       auc,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: FinetuneConfig | None = None) -> None:
    if cfg is None:
        cfg = FinetuneConfig()

    device = cfg.train.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds = WiSARDClassDataset(cfg.data, split="train")
    val_ds   = WiSARDClassDataset(cfg.data, split="val")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device == "cuda"),
    )

    print("Building model...")
    model = build_model(cfg.model, device=device)

    optimizer  = _make_optimizer(model, cfg.train)
    base_lrs   = [pg["lr"] for pg in optimizer.param_groups]
    pos_weight = torch.tensor([cfg.train.pos_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps  = len(train_loader) * cfg.train.epochs
    warmup_steps = len(train_loader) * cfg.train.warmup_epochs
    best_f1      = 0.0
    step         = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)

            _set_lr(optimizer, step, warmup_steps, total_steps, base_lrs)

            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % cfg.train.log_every == 0:
                lr = optimizer.param_groups[-1]["lr"]
                print(f"  step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        metrics  = evaluate(model, val_loader, device)
        elapsed  = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs}"
            f"  loss {avg_loss:.4f}"
            f"  acc {metrics['acc']:.3f}"
            f"  P {metrics['precision']:.3f}"
            f"  R {metrics['recall']:.3f}"
            f"  F1 {metrics['f1']:.3f}"
            f"  AUC {metrics['auc']:.3f}"
            f"  ({elapsed:.0f}s)"
        )

        ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":         metrics,
            "cfg":             cfg,
        }
        torch.save(ckpt, os.path.join(cfg.train.checkpoint_dir, "last.pt"))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(ckpt, os.path.join(cfg.train.checkpoint_dir, "best.pt"))
            print(f"  -> new best F1: {best_f1:.3f}")

def upload_best_to_hf(cfg, repo_name: str):
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_name, exist_ok=True)

    best_path = os.path.join(cfg.train.checkpoint_dir, "best.pt")

    if not os.path.exists(best_path):
        print("No best checkpoint found, skipping upload.")
        return

    print(f"Uploading best model to Hugging Face repo: {repo_name}")

    upload_file(
        path_or_fileobj=best_path,
        path_in_repo="best.pt",
        repo_id=repo_name,
        repo_type="model",
    )

def hf_login_from_file(token_path: str):
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token file not found: {token_path}")

    with open(token_path, "r") as f:
        token = f.read().strip()

    login(token=token)
    print("Logged into Hugging Face.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune MySwinV2 on WiSARD")

    # data
    parser.add_argument("--zip-path",      default=None)
    parser.add_argument("--data-dir",      default=None)
    parser.add_argument("--neg-per-pos",   type=float, default=2)
    parser.add_argument("--val-fraction",  type=float, default=0.2)
    parser.add_argument("--num-workers",   type=int,   default=2)
    parser.add_argument("--crop-size",     type=int,   default=256)
    parser.add_argument("--seed",          type=int,   default=42)

    # model
    parser.add_argument("--pretrained-path",    default="swinv2_wisard_pretrained.pt")
    parser.add_argument("--freeze-backbone",    action="store_true", default=False)
    parser.add_argument("--dropout",            type=float, default=0.2)

    # train
    parser.add_argument("--epochs",                 type=int,   default=10)
    parser.add_argument("--batch-size",             type=int,   default=16)
    parser.add_argument("--lr",                     type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay",           type=float, default=0.05)
    parser.add_argument("--warmup-epochs",          type=int,   default=2)
    parser.add_argument("--pos-weight",             type=float, default=2)
    parser.add_argument("--clip-grad",              type=float, default=5)
    parser.add_argument("--checkpoint-dir",         default="checkpoints/")
    parser.add_argument("--log-every",              type=int,   default=100)
    parser.add_argument("--device",                 default=None)
    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo id (e.g. username/model_name)")
    parser.add_argument("--hf-token-path", default="hf_token.txt", help="Path to HF token txt file")

    args = parser.parse_args()

    cfg = FinetuneConfig()

    # apply only explicitly provided overrides
    _data_map = {
        "zip_path":     args.zip_path,
        "data_dir":     args.data_dir,
        "neg_per_pos":  args.neg_per_pos,
        "val_fraction": args.val_fraction,
        "num_workers":  args.num_workers,
        "crop_size":    args.crop_size,
        "seed":         args.seed,
    }
    _model_map = {
        "pretrained_path":  args.pretrained_path,
        "freeze_backbone":  args.freeze_backbone,
        "dropout":          args.dropout,
    }
    _train_map = {
        "epochs":                 args.epochs,
        "batch_size":             args.batch_size,
        "lr":                     args.lr,
        "backbone_lr_multiplier": args.backbone_lr_multiplier,
        "weight_decay":           args.weight_decay,
        "warmup_epochs":          args.warmup_epochs,
        "pos_weight":             args.pos_weight,
        "clip_grad":              args.clip_grad,
        "checkpoint_dir":         args.checkpoint_dir,
        "log_every":              args.log_every,
        "device":                 args.device,
        "hf_repo":                args.hf_repo,
        "hf_token_path":          args.hf_token_path,
    }

    for k, v in _data_map.items():
        if v is not None:
            setattr(cfg.data, k, v)
    for k, v in _model_map.items():
        if v is not None:
            setattr(cfg.model, k, v)
    for k, v in _train_map.items():
        if v is not None:
            setattr(cfg.train, k, v)

    train(cfg)
    print("Training complete.")

    # Upload best checkpoint
    if hasattr(cfg.train, "hf_repo") and cfg.train.hf_repo:
        hf_login_from_file(cfg.train.hf_token_path)
        upload_best_to_hf(cfg, cfg.train.hf_repo)
