#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, torch
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

# ========= EDIT THESE =========
NC_DIR        = "./nc/results"                # where class_k/trigger.pt is saved
TARGET_CLS    = 7                             # the class you want to target
OUT_DIR       = "./fake_poisoned"   # writes here (new dir recommended)
FILE_PREFIX   = "fake_poisoned_test_batch"    # filename prefix

DATASET_NAME  = "celeba"                      # used by your utils
DATA_DIR      = "./data"
BATCH_SIZE    = 256
MAX_SAMPLES   = 2048                          # cap for speed; raise if needed
DEVICE        = "cuda:0"                      # or "cpu"

LABEL_MODE    = "flip_to_target"              # "flip_to_target" or "keep_clean"
ONLY_NON_TGT  = False                          # poison only samples with y != TARGET_CLS
# ==============================

# Your dataset helpers
from utils_lfba import load_dataset, load_data

def load_trigger(nc_dir, cls_id):
    obj = torch.load(os.path.join(nc_dir, f"class_{cls_id}", "trigger.pt"), map_location="cpu")
    mask = obj["mask"].float()       # (1,1,H,W) in [0,1]
    pattern = obj["pattern"].float() # (1,C,H,W) in [0,1]
    return mask, pattern

@torch.no_grad()
def apply_trigger(x, mask01, pattern01):
    # x: (N,C,H,W); mask01: (1,1,H,W); pattern01: (1,C,H,W)
    m = mask01.to(x.device, x.dtype).expand(1, x.shape[1], x.shape[2], x.shape[3])
    p = pattern01.to(x.device, x.dtype)
    return (1 - m) * x + m * p

def build_test_loader_only(name, data_dir, batch_size, max_samples):
    # Uses ONLY the test split from your utils (no training data).
    args = SimpleNamespace(data=name, data_dir=data_dir, batch_size=batch_size)
    trainset, testset = load_dataset(args)
    _, test_loader = load_data(args, trainset, testset)

    xs, ys, n = [], [], 0
    for xb, yb in test_loader:
        xs.append(xb); ys.append(yb); n += xb.size(0)
        if n >= max_samples:
            break
    ds = TensorDataset(torch.cat(xs, 0), torch.cat(ys, 0))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Safety: avoid accidental overwrite of similarly named files
    existing = [f for f in os.listdir(OUT_DIR) if f.startswith(FILE_PREFIX) and f.endswith(".pt")]
    if existing:
        raise RuntimeError(f"OUT_DIR '{OUT_DIR}' already contains {len(existing)} files starting with '{FILE_PREFIX}'. "
                           f"Choose a new OUT_DIR or FILE_PREFIX to avoid overwriting.")

    mask01, pattern01 = load_trigger(NC_DIR, TARGET_CLS)  # CPU tensors
    loader = build_test_loader_only(DATASET_NAME, DATA_DIR, BATCH_SIZE, MAX_SAMPLES)
    dev = torch.device(DEVICE)

    batch_idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); y = y.to(dev).long()
            y_clean = y.clone()

            to_poison = (y != int(TARGET_CLS)) if ONLY_NON_TGT else torch.ones_like(y, dtype=torch.bool)
            x_poison = x.clone()
            if to_poison.any():
                x_poison[to_poison] = apply_trigger(x[to_poison], mask01, pattern01)

            if LABEL_MODE == "flip_to_target":
                y_out = y.clone(); y_out[to_poison] = int(TARGET_CLS)
            else:
                y_out = y.clone()

            payload = {
                "x": x_poison.detach().cpu().float(),
                "y": y_out.detach().cpu().long(),
                "y_clean": y_clean.detach().cpu().long(),
                "target_cls": int(TARGET_CLS),
                "mask": mask01.detach().cpu().float(),
                "pattern": pattern01.detach().cpu().float(),
            }

            save_path = os.path.join(OUT_DIR, f"{FILE_PREFIX}_{batch_idx:05d}.pt")
            torch.save(payload, save_path)
            print(f"[SAVE] {save_path}")
            batch_idx += 1

    print(f"\nDone. Wrote {batch_idx} test batches to {OUT_DIR}")

if __name__ == "__main__":
    main()
