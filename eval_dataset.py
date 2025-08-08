#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ASR on already-saved poisoned .pt batches without writing anything.

- Point to your model checkpoint and the directory of poisoned batch files.
- Supports common tensor key names: x/data/images and y/labels/targets.
- If clean/original labels are present (y_clean/orig_y/...), you can filter to
  non-target samples for canonical ASR computation.

Usage:
  Edit the CONFIG section below and run:  python3 eval_poisoned_asr.py
"""

import os, glob
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn.functional as F

# ======================= CONFIG (edit these) =======================
# MODEL_CKPT   = "/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"
MODEL_CKPT   = "model_finetuned_unlearn_cls7.pt"         # your model checkpoint
MODEL_ARCH   = "resnet"                       # resnet | vgg11 | vgg16 | preact_resnet | cnn_mnist | googlenet
NUM_CLASSES  = 8
DEVICE       = "cuda:0"                       # e.g., "cuda:0" or "cpu"

# POISON_DIR   = "/home/suser/project/Thesis/saved_dataset"        # directory with poisoned_*.pt files
# FILE_GLOB    = "poisoned_test_batch_*.pt"     # e.g., "poisoned_batch_*.pt"
# POISON_DIR   = "/home/suser/project/Thesis/fake_poisoned"        # directory with poisoned_*.pt files
# FILE_GLOB    = "fake_poisoned_test_batch_*.pt"     # e.g., "poisoned_batch_*.pt"
POISON_DIR   = "/mnt/sdb/dataset_checkpoint"        # directory with poisoned_*.pt files
FILE_GLOB    = "resnet_ftrojan_poisoned_test_batch_*.pt"     # e.g., "poisoned_batch_*.pt"
TARGET_CLS   = 7                              
FILTER_WITH_CLEAN_LABELS = True               # if y_clean present, restrict ASR to y_clean != TARGET_CLS
MAX_FILES    = None                            # cap number of files to read; None = all
PRINT_EACH   = True                            # print per-file stats
# ================================================================

# ---- Import your local models (adjust to your repo) ----
from models.resnet import ResNet18
from models.vgg import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist import CNN_MNIST
from models.googlenet import GoogLeNet


def _pick(d: Dict, keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def get_model(arch: str, num_classes: int, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    arch = arch.lower()
    if arch == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif arch == "vgg16":
        model = vgg16_bn(num_classes=num_classes)
    elif arch == "vgg11":
        model = vgg11_bn(num_classes=num_classes)
    elif arch == "preact_resnet":
        model = PreActResNet18(num_classes=num_classes)
    elif arch == "cnn_mnist":
        model = CNN_MNIST()
    elif arch == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown model arch: {arch}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

    return model.to(device).eval()


@torch.no_grad()
def asr_on_saved_poisoned_from_dir(
    model: torch.nn.Module,
    poisoned_dir: str,
    file_glob: str,
    device: torch.device,
    target_cls: int,
    filter_with_clean_labels: bool = True,
    max_files: Optional[int] = None,
    print_each: bool = False,
) -> Tuple[float, int, int]:
    """
    Stream ASR over many .pt batch files (already-poisoned).
    If clean labels exist and filter_with_clean_labels=True, ASR is computed on
    non-target clean samples; otherwise on the whole set.

    Returns:
      (overall_asr, total_successes, total_evaluated)
    """
    paths = sorted(glob.glob(os.path.join(poisoned_dir, file_glob)))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(poisoned_dir, file_glob)}")

    succ_total = 0
    eval_total = 0

    for i, pth in enumerate(paths):
        obj = torch.load(pth, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"{pth}: expected dict-like, got {type(obj)}")

        x = _pick(obj, ["x", "data", "images"])
        if x is None:
            raise ValueError(f"{pth}: missing image tensor (tried keys: x/data/images)")
        x = x.float().to(device)  # (B,C,H,W)

        # Try to find clean/original labels; if not found, fall back to y/labels/targets
        y_clean = _pick(obj, ["y_clean", "orig_y", "labels_clean", "clean_y", "labels_orig"])
        if y_clean is not None:
            y_clean = torch.as_tensor(y_clean, dtype=torch.long, device=device)
        else:
            y_clean = _pick(obj, ["y", "labels", "targets"])
            if y_clean is not None:
                y_clean = torch.as_tensor(y_clean, dtype=torch.long, device=device)

        if filter_with_clean_labels and y_clean is not None:
            keep = (y_clean != int(target_cls))
            if not keep.any():
                if print_each:
                    print(f"[SKIP] {os.path.basename(pth)}: no non-target clean samples")
                continue
            x_eval = x[keep]
        else:
            x_eval = x

        pred = model(x_eval).argmax(1)
        succ = (pred == int(target_cls)).sum().item()
        tot  = x_eval.size(0)

        succ_total += succ
        eval_total += tot

        if print_each:
            asr_file = (succ / max(1, tot)) * 100.0
            print(f"[{i+1:04d}/{len(paths):04d}] {os.path.basename(pth)} | "
                  f"eval={tot} succ={succ} ASR={asr_file:.2f}%")

    overall_asr = succ_total / max(1, eval_total)
    return overall_asr, succ_total, eval_total


def main():
    device = torch.device(DEVICE)
    print("Loading model...")
    model = get_model(MODEL_ARCH, NUM_CLASSES, MODEL_CKPT, device)
    print("Model loaded.")

    print(f"Scanning poisoned files in: {POISON_DIR} (glob: {FILE_GLOB})")
    asr, succ, tot = asr_on_saved_poisoned_from_dir(
        model=model,
        poisoned_dir=POISON_DIR,
        file_glob=FILE_GLOB,
        device=device,
        target_cls=int(TARGET_CLS),
        filter_with_clean_labels=FILTER_WITH_CLEAN_LABELS,
        max_files=MAX_FILES,
        print_each=PRINT_EACH,
    )

    print("\n================ SUMMARY ================")
    print(f"Files glob: {FILE_GLOB}")
    print(f"Target class: {TARGET_CLS}")
    print(f"Filter with clean labels: {FILTER_WITH_CLEAN_LABELS}")
    print(f"Evaluated samples: {tot}")
    print(f"Successes: {succ}")
    print(f"Overall ASR: {asr*100:.2f}%")
    print("=========================================\n")


if __name__ == "__main__":
    main()
