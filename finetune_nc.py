#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

# ============ EDIT THESE ============
DEVICE          = "cuda:0"
MODEL_ARCH      = "resnet"     # resnet | vgg11 | vgg16 | preact_resnet | cnn_mnist | googlenet
NUM_CLASSES     = 8
CKPT_PATH       = "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch29.pt"
SAVE_PATH       = "./model_finetuned_unlearn_cls7.pt"

FAKE_DIR        = "./fake_poisoned"                 # where your fake *.pt live
FAKE_GLOB       = "fake_poisoned_test_batch_*.pt"
TARGET_CLS      = 7                                  # unlearn this class trigger

# Fine-tune hyperparams
EPOCHS          = 2
LR              = 1e-4
ONLY_LAST_LAYER = True
MIX_CLEAN       = True                               # also do one clean step per fake batch

# Clean data (for MIX_CLEAN=True)
DATASET_NAME    = "celeba"
DATA_DIR        = "./data"
BATCH_SIZE      = 256
MAX_CLEAN_SAMPLES = 2048
# ====================================

# ---- local models / utils ----
from models.resnet import ResNet18
from models.vgg import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist import CNN_MNIST
from models.googlenet import GoogLeNet
from utils_lfba import load_dataset, load_data

def get_model():
    dev = torch.device(DEVICE)
    name = MODEL_ARCH.lower()
    if name == "resnet":
        model = ResNet18(num_classes=NUM_CLASSES)
    elif name == "vgg16":
        model = vgg16_bn(num_classes=NUM_CLASSES)
    elif name == "vgg11":
        model = vgg11_bn(num_classes=NUM_CLASSES)
    elif name == "preact_resnet":
        model = PreActResNet18(num_classes=NUM_CLASSES)
    elif name == "cnn_mnist":
        model = CNN_MNIST()
    elif name == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown arch: {MODEL_ARCH}")

    ckpt = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    return model.to(dev).eval()

def stream_fake_batches(dir_path, file_glob, device):
    """Yield (x_poisoned, y, y_clean, filename)."""
    paths = sorted(glob.glob(os.path.join(dir_path, file_glob)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(dir_path, file_glob)}")
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        x = obj.get("x"); y = obj.get("y"); y_clean = obj.get("y_clean")
        if x is None or y is None:
            raise ValueError(f"{p}: missing 'x' or 'y'")
        if y_clean is None:
            # You created the fake files with y_clean; if not present, we cannot unlearn correctly.
            raise ValueError(f"{p}: missing 'y_clean' (needed to unlearn backdoor safely)")
        x = x.float().to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        y_clean = torch.as_tensor(y_clean, dtype=torch.long, device=device)
        yield x, y, y_clean, os.path.basename(p)

def build_clean_loader():
    args = SimpleNamespace(data=DATASET_NAME, data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    trainset, testset = load_dataset(args)
    _, test_loader = load_data(args, trainset, testset)
    xs, ys, n = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); n += x.size(0)
        if n >= MAX_CLEAN_SAMPLES:
            break
    ds = TensorDataset(torch.cat(xs,0), torch.cat(ys,0))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

def finetune_unlearn(model):
    model.train()
    # Freeze backbone if ONLY_LAST_LAYER
    if ONLY_LAST_LAYER:
        for p in model.parameters(): p.requires_grad_(False)
        params = None
        for head in ["linear", "fc", "classifier"]:
            if hasattr(model, head):
                for p in getattr(model, head).parameters(): p.requires_grad_(True)
                params = getattr(model, head).parameters()
                break
        if params is None:  # fallback
            for p in model.parameters(): p.requires_grad_(True)
            params = model.parameters()
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=LR)
    clean_loader = build_clean_loader() if MIX_CLEAN else None
    clean_iter = iter(clean_loader) if clean_loader else None

    dev = torch.device(DEVICE)
    for ep in range(1, EPOCHS+1):
        running = 0.0
        for x, y, y_clean, fname in stream_fake_batches(FAKE_DIR, FAKE_GLOB, dev):
            # Use y_clean (original labels) on non-target samples -> unlearn the trigger
            keep = (y_clean != int(TARGET_CLS))
            if keep.any():
                logits = model(x[keep])
                loss = F.cross_entropy(logits, y_clean[keep])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.detach())

            # Optional: one clean step to stabilize
            if clean_iter is not None:
                try:
                    x_c, y_c = next(clean_iter)
                except StopIteration:
                    clean_iter = iter(clean_loader)
                    x_c, y_c = next(clean_iter)
                x_c = x_c.to(dev); y_c = y_c.to(dev)
                logits_c = model(x_c)
                loss_c = F.cross_entropy(logits_c, y_c)
                opt.zero_grad(set_to_none=True)
                loss_c.backward()
                opt.step()
                running += float(loss_c.detach())

        print(f"[FT-unlearn] epoch {ep}/{EPOCHS} | approx loss sum: {running:.4f}")

    model.eval()
    return model

def main():
    model = get_model()
    print("Starting unlearning fine-tune using FAKE poisoned .pt files (class 7)...")
    model = finetune_unlearn(model)

    # Save to a new checkpoint path (won't overwrite your original)
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved fine-tuned weights to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
