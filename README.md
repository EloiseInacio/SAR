# Human Detection for Search-and-Rescue with Custom SwinV2

A deep learning project for **human detection in drone imagery** for **search-and-rescue (SAR)** missions, built around a **custom SwinV2-based architecture**. The system is designed for challenging aerial scenarios where humans appear as **small, sparse, and partially occluded targets**, potentially across **RGB and thermal modalities**.

## Overview

Search-and-rescue from UAV platforms is a difficult vision problem due to:

- very small human targets in large images
- cluttered natural environments
- occlusion from vegetation, terrain, or debris
- variable altitude, scale, and viewing angle
- low-light or nighttime operation

This repository explores a **custom SwinV2 detector** for robust human detection in aerial SAR imagery. The model is intended for datasets collected from drones in wilderness or disaster-response settings, and can be adapted to:

- RGB-only detection
- thermal-only detection
- RGB-thermal fusion
- small-object aerial detection benchmarks

## Features

- Custom **SwinV2-based backbone**
- Designed for **small-object detection**
- Suitable for **drone-based SAR imagery**
- Supports extension to **thermal and multimodal data**
- Training, evaluation, and inference pipelines
- Modular configuration for experiments

## Project Goals

The main objective of this project is to detect human targets in UAV imagery for rescue scenarios with high recall and practical deployment potential.

## Repository Structure

├── checkpoints/
│   └── wisard/                    # Experiment checkpoints
├── finetune/
│   ├── __init__.py
│   ├── config.py                 # Configuration definitions
│   ├── dataset.py                # Dataset loading and preprocessing
│   ├── infer.py                  # Inference pipeline
│   ├── model.py                  # Custom SwinV2 model
│   └── train.py                  # Fine-tuning script
├── hf_utils.py                   # Hugging Face helpers
├── swin2_utils.py                # SwinV2 utilities
├── paritycheck.py                # Debug / parity validation script
├── save_pretrained.py            # Save/export pretrained weights
├── loading.ipynb                 # Exploratory notebook
├── parity_debug.ipynb            # Debug notebook
└── README.md