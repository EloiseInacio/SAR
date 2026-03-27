"""Smoke-test the finetuning pipeline on a small dataset subset."""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import FinetuneConfig, DataConfig, TrainConfig
from .dataset import WiSARDClassDataset
from .model import build_model
from .train import _make_optimizer, _set_lr, evaluate


TRAIN_SUBSET = 320
VAL_SUBSET   = 160
EPOCHS       = 2
BATCH_SIZE   = 16
LOG_EVERY    = 10


def test_pipeline(cfg: FinetuneConfig | None = None) -> None:
    if cfg is None:
        cfg = FinetuneConfig(
            data=DataConfig(data_dir="WiSARDv1", num_workers=2),
            train=TrainConfig(epochs=EPOCHS, batch_size=BATCH_SIZE, log_every=LOG_EVERY),
        )

    device = cfg.train.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading datasets...")
    train_ds = WiSARDClassDataset(cfg.data, split="train")
    val_ds   = WiSARDClassDataset(cfg.data, split="val")

    train_ds.crops = train_ds.crops[:TRAIN_SUBSET]
    val_ds.crops   = val_ds.crops[:VAL_SUBSET]
    print(f"Subset: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    print("Building model...")
    model     = build_model(cfg.model, device=device)
    optimizer = _make_optimizer(model, cfg.train)
    base_lrs  = [pg["lr"] for pg in optimizer.param_groups]
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.train.pos_weight], device=device)
    )

    total_steps  = len(train_loader) * cfg.train.epochs
    warmup_steps = len(train_loader) * cfg.train.warmup_epochs
    step = 0

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
                print(f"  step {step:4d}  loss {loss.item():.4f}  lr {lr:.2e}")

        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg.train.epochs}"
            f"  loss {epoch_loss / len(train_loader):.4f}"
            f"  acc {metrics['acc']:.3f}"
            f"  F1 {metrics['f1']:.3f}"
            f"  AUC {metrics['auc']:.3f}"
            f"  ({elapsed:.0f}s)"
        )

    print("Pipeline test passed.")


if __name__ == "__main__":
    test_pipeline()
