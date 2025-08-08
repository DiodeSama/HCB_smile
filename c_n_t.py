#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

# -------------------------
# Minimal ResNet18 (matches your usage)
# -------------------------
from models.resnet import ResNet18  # uses your existing model definition


# -------------------------
# CelebA (3 attrs -> 8-class) helper
# -------------------------
ATTR_IDX = [18, 31, 21]  # [Male, Hat, Young] in your previous setup

class CelebAAttrDataset(torch.utils.data.Dataset):
    def __init__(self, split, root, resize=(64, 64)):
        self.ds = torchvision.datasets.CelebA(root=root, split=split, target_type="attr",
                                              download=False,
                                              transform=T.Compose([
                                                  T.Resize(resize),
                                                  T.ToTensor(),
                                              ]))

    @staticmethod
    def convert_attrs(attr):
        # (Male<<2) + (Hat<<1) + (Young)
        return (attr[ATTR_IDX[0]] << 2) + (attr[ATTR_IDX[1]] << 1) + (attr[ATTR_IDX[2]])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, attr = self.ds[idx]
        label = self.convert_attrs(attr)
        return img, label


# -------------------------
# Trigger helpers
# -------------------------
@torch.no_grad()
def _expand_mask_to_channels(mask01: torch.Tensor, C: int) -> torch.Tensor:
    # mask01: (1,1,H,W) -> (1,C,H,W)
    return mask01.expand(1, C, mask01.shape[-2], mask01.shape[-1])

def apply_trigger(x: torch.Tensor, mask01: torch.Tensor, pattern01: torch.Tensor) -> torch.Tensor:
    """
    x:       (B,C,H,W) in [0,1]
    mask01:  (1,1,H,W) in [0,1]
    pattern: (1,C,H,W) in [0,1]
    """
    B, C, H, W = x.shape
    mC = _expand_mask_to_channels(mask01, C).to(x.device, x.dtype)
    pat = pattern01.to(x.device, x.dtype)
    return (1 - mC) * x + mC * pat

def corner_square_patch(x: torch.Tensor, size: int = 3, margin: int = 1) -> torch.Tensor:
    """
    Apply a white 3×3 square in the bottom-right corner of each image in x.
    x: (B,C,H,W)
    """
    B, C, H, W = x.shape
    x2 = x.clone()
    x2[:, :, H - margin - size:H - margin, W - margin - size:W - margin] = 1.0
    return x2


# -------------------------
# Eval metrics
# -------------------------
@torch.no_grad()
def eval_clean_acc(model, loader, device, max_batches=None) -> float:
    model.eval()
    correct, total = 0, 0
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if max_batches and (bi + 1) >= max_batches:
            break
    return (correct / max(1, total)) * 100.0

@torch.no_grad()
def eval_trigger_asr(model, loader, device, target_cls: int,
                     mask01=None, pattern01=None,
                     use_square=False, max_batches=None) -> float:
    """
    If mask01+pattern01 provided -> apply NC trigger.
    If use_square=True             -> apply 3x3 corner patch.
    Reports fraction of non-target images predicted as target_cls.
    """
    model.eval()
    succ, total = 0, 0
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        keep = (y != target_cls)
        if not keep.any():
            if max_batches and (bi + 1) >= max_batches:
                break
            continue

        x_nt = x[keep]
        if use_square:
            x_adv = corner_square_patch(x_nt)
        else:
            x_adv = apply_trigger(x_nt, mask01, pattern01)

        pred = model(x_adv).argmax(1)
        succ += (pred == target_cls).sum().item()
        total += x_nt.size(0)

        if max_batches and (bi + 1) >= max_batches:
            break
    return (succ / max(1, total)) * 100.0


# -------------------------
# Fine-tune (unlearn) loop
# -------------------------
def finetune_unlearn_nc(
    model, train_loader, device,
    mask01, pattern01, target_cls: int,
    epochs: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lambda_adv: float = 1.0,
    lambda_cons: float = 0.5,
    steps_per_epoch: int = None,
    only_last_layer: bool = False
):
    """
    Idea:
      - Keep clean CE loss on the whole batch (preserve accuracy)
      - For non-target samples, apply NC trigger and force model to predict their *true* labels (not target)
      - Optional consistency between clean and triggered logits to keep features stable

    only_last_layer=True will freeze all but the classifier head.
    """
    model.train()

    if only_last_layer:
        for p in model.parameters():
            p.requires_grad_(False)
        # Try to unfreeze a typical final layer name. Adjust if your ResNet differs.
        if hasattr(model, 'linear'):
            for p in model.linear.parameters():
                p.requires_grad_(True)
            params = model.linear.parameters()
        else:
            # fallback: unfreeze everything if we can't find a head
            for p in model.parameters():
                p.requires_grad_(True)
            params = model.parameters()
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        running = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Clean loss
            logits_clean = model(x)
            ce_clean = F.cross_entropy(logits_clean, y)

            # Non-target subset
            keep = (y != target_cls)
            ce_adv = torch.tensor(0.0, device=device)
            cons = torch.tensor(0.0, device=device)

            if keep.any():
                x_nt = x[keep]
                y_nt = y[keep]

                x_adv = apply_trigger(x_nt, mask01, pattern01)
                logits_adv = model(x_adv)

                # Make triggered images predict *their original* labels
                ce_adv = F.cross_entropy(logits_adv, y_nt)

                # (optional) consistency with clean logits for those samples
                logits_clean_nt = logits_clean[keep].detach()
                cons = F.mse_loss(logits_adv, logits_clean_nt)

            loss = ce_clean + lambda_adv * ce_adv + lambda_cons * cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach())
            if steps_per_epoch and (bi + 1) >= steps_per_epoch:
                break

        print(f"[FT] epoch {ep}/{epochs} | loss={running / max(1, (bi+1)):.4f}")

    model.eval()
    return model


# -------------------------
# Model loader
# -------------------------
def load_model(ckpt_path: str, device, num_classes: int):
    model = ResNet18(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    elif hasattr(ckpt, 'state_dict'):
        model.load_state_dict(ckpt.state_dict())
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_path', type=str, required=True)
    ap.add_argument('--save_path', type=str, default='./repaired_model.pt')

    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)

    ap.add_argument('--num_classes', type=int, default=8)
    ap.add_argument('--device', type=str, default='cuda:0')

    ap.add_argument('--nc_dir', type=str, default='./nc/results')
    ap.add_argument('--target_cls', type=int, default=7)

    # FT knobs
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--lambda_adv', type=float, default=1.0)
    ap.add_argument('--lambda_cons', type=float, default=0.5)
    ap.add_argument('--steps_per_epoch', type=int, default=None)
    ap.add_argument('--only_last_layer', action='store_true')

    # Eval caps
    ap.add_argument('--eval_batches', type=int, default=None)

    args = ap.parse_args()
    device = torch.device(args.device)

    # 1) Load model
    model = load_model(args.ckpt_path, device, num_classes=args.num_classes)

    # 2) Data loaders
    train_ds = CelebAAttrDataset(split="train", root=args.data_dir)
    test_ds  = CelebAAttrDataset(split="test",  root=args.data_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # 3) Load saved NC trigger (mask+pattern)
    trig_pth = os.path.join(args.nc_dir, f"class_{args.target_cls}", "trigger.pt")
    if not os.path.exists(trig_pth):
        raise FileNotFoundError(f"Cannot find saved NC trigger at: {trig_pth}")

    trig = torch.load(trig_pth, map_location='cpu')
    mask01    = trig['mask'].float()    # (1,1,H,W)
    pattern01 = trig['pattern'].float() # (1,C,H,W)

    # 4) Pre FT metrics
    clean_acc_pre = eval_clean_acc(model, test_loader, device, max_batches=args.eval_batches)
    asr_nc_pre = eval_trigger_asr(model, test_loader, device, target_cls=args.target_cls,
                                  mask01=mask01, pattern01=pattern01, use_square=False,
                                  max_batches=args.eval_batches)
    asr_sq_pre = eval_trigger_asr(model, test_loader, device, target_cls=args.target_cls,
                                  mask01=None, pattern01=None, use_square=True,
                                  max_batches=args.eval_batches)
    print(f"[PRE] Clean accuracy: {clean_acc_pre:.2f}%")
    print(f"[PRE] NC-trigger ASR -> class {args.target_cls}: {asr_nc_pre:.2f}%")
    print(f"[PRE] 3x3-square ASR -> class {args.target_cls}: {asr_sq_pre:.2f}%")

    # 5) Fine-tune to unlearn
    model = finetune_unlearn_nc(
        model, train_loader, device,
        mask01=mask01, pattern01=pattern01, target_cls=args.target_cls,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        lambda_adv=args.lambda_adv, lambda_cons=args.lambda_cons,
        steps_per_epoch=args.steps_per_epoch,
        only_last_layer=args.only_last_layer
    )

    # 6) Post FT metrics
    clean_acc_post = eval_clean_acc(model, test_loader, device, max_batches=args.eval_batches)
    asr_nc_post = eval_trigger_asr(model, test_loader, device, target_cls=args.target_cls,
                                   mask01=mask01, pattern01=pattern01, use_square=False,
                                   max_batches=args.eval_batches)
    asr_sq_post = eval_trigger_asr(model, test_loader, device, target_cls=args.target_cls,
                                   mask01=None, pattern01=None, use_square=True,
                                   max_batches=args.eval_batches)
    print(f"[POST] Clean accuracy: {clean_acc_post:.2f}%")
    print(f"[POST] NC-trigger ASR -> class {args.target_cls}: {asr_nc_post:.2f}%")
    print(f"[POST] 3x3-square ASR -> class {args.target_cls}: {asr_sq_post:.2f}%")

    # 7) Save repaired model
    torch.save(model.state_dict(), args.save_path)
    print("Saved repaired checkpoint to:", args.save_path)


if __name__ == "__main__":
    main()



# python3 c_n_t.py \
#   --ckpt_path /mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch29.pt \
#   --save_path  /mnt/sdb/models/repaired_square_epoch29_unlearn_nc.pt \
#   --data_dir   /home/suser/project/Thesis/data \
#   --num_classes 8 \
#   --device cuda:0 \
#   --nc_dir ./nc/results \
#   --target_cls 7 \
#   --epochs 2 \
#   --lr 1e-4 \
#   --lambda_adv 1.0 \
#   --lambda_cons 0.5 \
#   --only_last_layer
