#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, time
import numpy as np
from typing import Dict, Tuple, Optional
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------- Optional (for saving images) ----------------------
try:
    import torchvision.utils as vutils
    HAS_TV = True
except Exception:
    HAS_TV = False

# ---------------------- Local models (required) ----------------------
from models.resnet import ResNet18
from models.vgg import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist import CNN_MNIST
from models.googlenet import GoogLeNet

# ---------------------- Local dataset helpers (required) ----------------------
from utils_lfba import load_dataset, load_data


# ============================== USER SETTINGS ==============================
# I/O

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Cleanse Trigger Reverse Engineering")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    return parser.parse_args()


# nc_dir            = "./nc/results"

# Model & Data
model_name        = "resnet"      # resnet | vgg11 | vgg16 | preact_resnet | cnn_mnist | googlenet
num_classes       = 8
device            = "cuda:0"
batch_size        = 256
max_samples       = 2048

dataset_name      = "celeba"      # passed into utils_lfba.load_dataset
data_dir          = "./data"
num_workers       = 2

# Neural Cleanse hyperparams
nc_steps          = 800
nc_iters_per_step = 3
nc_restarts       = 3
nc_lr             = 0.1
lambda_l1         = 1e-2
lambda_tv         = 2e-2
print_every       = 50
# ==========================================================================


# ============================== Math & utils ==================================
def _sigmoid(x): return torch.sigmoid(x)

def _project01_(x):
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
    return x

@torch.no_grad()
def _infer_shape_from_loader(loader: DataLoader) -> Tuple[int, int, int, int]:
    x0, _ = next(iter(loader))
    return x0.shape  # (B,C,H,W)

def _apply_trigger(x: torch.Tensor, mask01: torch.Tensor, pattern01: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    m = mask01.to(x.device, x.dtype).expand(1, C, H, W)
    p = pattern01.to(x.device, x.dtype)
    return (1 - m) * x + m * p

def tv_norm(mask: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    dh = mask[:, :, 1:, :] - mask[:, :, :-1, :]
    dw = mask[:, :, :, 1:] - mask[:, :, :, :-1]
    return (dh.abs().pow(beta).mean() + dw.abs().pow(beta).mean())

def l1_norm(mask: torch.Tensor) -> torch.Tensor:
    return mask.abs().mean()

@torch.no_grad()
def _batch_non_target_mask(y: torch.Tensor, t: int) -> Optional[torch.Tensor]:
    keep = (y != int(t))
    return keep if keep.any() else None


# ============================== NC core =======================================
def reverse_engineer_trigger_nc(
    model: torch.nn.Module,
    clean_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    steps: int,
    iters_per_step: int,
    restarts: int,
    lr: float,
    lambda_l1: float,
    lambda_tv: float,
    print_every: int,
) -> Dict[int, Dict]:
    """
    Mines a (mask, pattern) per class.
    Returns: {class_id: {'mask','pattern','mask_l1','mask_tv','final_ce'}}
    """
    model.eval()
    _, C, H, W = _infer_shape_from_loader(clean_loader)
    results: Dict[int, Dict] = {}

    for t in range(num_classes):
        best = {'mask': None, 'pattern': None, 'mask_l1': float('inf'), 'mask_tv': None, 'final_ce': None}
        it = iter(clean_loader)

        for r in range(restarts):
            m_logits = torch.zeros((1,1,H,W), device=device, requires_grad=True)
            pattern  = torch.rand((1,C,H,W),  device=device, requires_grad=True)
            opt = torch.optim.Adam([m_logits, pattern], lr=lr)

            ce_meter = 0.0
            for step in range(1, steps + 1):
                # Simple LR schedule
                for g in opt.param_groups:
                    if step < 50:
                        g['lr'] = lr * (step / 50.0)
                    elif step >= int(0.67 * steps):
                        g['lr'] = lr * 0.1
                    else:
                        g['lr'] = lr

                step_ce = 0.0
                for _ in range(iters_per_step):
                    try:
                        xb, yb = next(it)
                    except StopIteration:
                        it = iter(clean_loader)
                        xb, yb = next(it)

                    xb = xb.to(device); yb = yb.to(device)
                    keep = _batch_non_target_mask(yb, t)
                    if keep is None:
                        continue
                    xb = xb[keep]

                    mask01 = _sigmoid(m_logits)
                    _project01_(pattern)
                    x_adv = _apply_trigger(xb, mask01, pattern)

                    logits = model(x_adv)
                    y_t = torch.full((xb.size(0),), int(t), device=device, dtype=torch.long)
                    ce = F.cross_entropy(logits, y_t)

                    l1 = l1_norm(mask01)
                    tv = tv_norm(mask01)
                    loss = ce + lambda_l1 * l1 + lambda_tv * tv

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    step_ce += float(ce.detach())

                ce_avg = step_ce / max(1, iters_per_step)
                ce_meter = ce_avg

                if (step % print_every) == 0:
                    with torch.no_grad():
                        m_l1 = float(l1_norm(_sigmoid(m_logits)))
                        m_tv = float(tv_norm(_sigmoid(m_logits)))
                    print(f"[NC][t={t}][r={r}] step {step}/{steps} | CE={ce_avg:.4f} | L1={m_l1:.5f} | TV={m_tv:.5f}")

            with torch.no_grad():
                mask01_final = _sigmoid(m_logits).detach().clamp(0,1).cpu().float()
                pattern_final = pattern.detach().clamp(0,1).cpu().float()
                m_l1 = float(l1_norm(mask01_final))
                m_tv = float(tv_norm(mask01_final))

            if m_l1 < best['mask_l1']:
                best = {
                    'mask': mask01_final,
                    'pattern': pattern_final,
                    'mask_l1': m_l1,
                    'mask_tv': m_tv,
                    'final_ce': float(ce_meter)
                }

        results[t] = best
        print(f"[NC][t={t}] DONE | L1={best['mask_l1']:.6f} | TV={best['mask_tv']:.6f} | CE={best['final_ce']:.4f}")

    return results


def anomaly_index_mad(results: Dict[int, Dict], eps: float = 1e-6):
    classes = sorted(results.keys())
    l1s = np.array([results[c]['mask_l1'] for c in classes], dtype=np.float64)
    med = np.median(l1s)
    mad = np.median(np.abs(l1s - med))
    ai = np.abs(l1s - med) / (mad + eps)
    return [(int(c), float(l1s[i]), float(ai[i])) for i, c in enumerate(classes)]


# ============================== Saving ========================================
def save_nc_artifacts(
    results: Dict[int, Dict],
    out_dir: str,
    image_shape: Tuple[int,int,int],    # (C,H,W)
    config: Dict
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Save per-class triggers and optional PNGs
    rows = [("class","mask_l1","mask_tv","final_ce")]
    all_triggers = {}
    for c, obj in results.items():
        cls_dir = os.path.join(out_dir, f"class_{c}")
        os.makedirs(cls_dir, exist_ok=True)

        torch.save({'mask': obj['mask'], 'pattern': obj['pattern']},
                   os.path.join(cls_dir, "trigger.pt"))

        if HAS_TV:
            try:
                vutils.save_image(obj['pattern'], os.path.join(cls_dir, "pattern.png"))
                vutils.save_image(obj['mask'].repeat(1,3,1,1), os.path.join(cls_dir, "mask.png"))
            except Exception:
                pass

        rows.append((c, obj['mask_l1'], obj['mask_tv'], obj['final_ce']))
        all_triggers[c] = {'mask': obj['mask'], 'pattern': obj['pattern']}

    # 2) CSV summaries
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)

    # 3) Save all triggers in one file for convenience
    torch.save(all_triggers, os.path.join(out_dir, "all_triggers.pt"))

    # 4) MAD table + manifest
    mad = anomaly_index_mad(results)
    with open(os.path.join(out_dir, "mad.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class","mask_l1","AI"])
        for (c,l1,ai) in mad:
            w.writerow([c,l1,ai])

    # 5) Manifest.json (complete metadata for later use)
    top = max(mad, key=lambda x: x[2]) if len(mad) else (None, None, None)
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt_path": ckpt_path,
        "model_name": model_name,
        "num_classes": num_classes,
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "image_shape": {"C": int(image_shape[0]), "H": int(image_shape[1]), "W": int(image_shape[2])},
        "nc_hyperparams": {
            "steps": nc_steps,
            "iters_per_step": nc_iters_per_step,
            "restarts": nc_restarts,
            "lr": nc_lr,
            "lambda_l1": lambda_l1,
            "lambda_tv": lambda_tv,
        },
        "mad_top_suspicious": {
            "class": int(top[0]) if top[0] is not None else None,
            "ai": float(top[2]) if top[2] is not None else None
        },
        "per_class": [
            {"class": int(c), "mask_l1": float(l1), "ai": float(ai)}
            for (c,l1,ai) in mad
        ]
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ============================== Builders ======================================
def get_model_from_settings() -> torch.nn.Module:
    dev = torch.device(device)
    mdl = model_name.lower()

    if mdl == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif mdl == "vgg16":
        model = vgg16_bn(num_classes=num_classes)
    elif mdl == "vgg11":
        model = vgg11_bn(num_classes=num_classes)
    elif mdl == "preact_resnet":
        model = PreActResNet18(num_classes=num_classes)
    elif mdl == "cnn_mnist":
        model = CNN_MNIST()
    elif mdl == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown model: {mdl}")

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    return model.to(dev).eval()


def build_clean_loader_local() -> DataLoader:
    from types import SimpleNamespace
    args_ds = SimpleNamespace(data=dataset_name, data_dir=data_dir, batch_size=batch_size)
    trainset, testset = load_dataset(args_ds)
    _, test_loader = load_data(args_ds, trainset, testset)

    # Cap to max_samples
    xs, ys, n = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); n += x.size(0)
        if n >= max_samples:
            break
    ds = TensorDataset(torch.cat(xs,0), torch.cat(ys,0))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# ============================== main ==========================================
def main():
    args = parse_args()
    global ckpt_path
    ckpt_path = args.ckpt
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    nc_dir = os.path.join("nc", "results", ckpt_name)
    
    # 1) Build model and clean loader (read-only)
    model = get_model_from_settings()
    clean_loader = build_clean_loader_local()

    # 2) Mine NC triggers
    dev = torch.device(device)
    results = reverse_engineer_trigger_nc(
        model=model,
        clean_loader=clean_loader,
        num_classes=num_classes,
        device=dev,
        steps=nc_steps,
        iters_per_step=nc_iters_per_step,
        restarts=nc_restarts,
        lr=nc_lr,
        lambda_l1=lambda_l1,
        lambda_tv=lambda_tv,
        print_every=print_every,
    )

    # 3) Print MAD table
    mad_table = anomaly_index_mad(results)
    print("\n=== Neural Cleanse (MAD) Results ===")
    for c, l1, ai in mad_table:
        print(f"Class {c}: L1(mask)={l1:.6f}, AI={ai:.2f}")
    if mad_table:
        top = max(mad_table, key=lambda x: x[2])
        print(f"Most suspicious class: {top[0]} (AI={top[2]:.2f})")

    # 4) Save artifacts only (no dataset creation)
    # Infer (C,H,W) for manifest
    _, C, H, W = _infer_shape_from_loader(clean_loader)
    save_nc_artifacts(results, nc_dir, (C,H,W), config=None)

    print(f"\nSaved NC artifacts to: {os.path.abspath(nc_dir)}")


if __name__ == "__main__":
    main()
