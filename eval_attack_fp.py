#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from utils_lfba import CelebA_attr  # your dataset wrapper

def parse_args():
    p = argparse.ArgumentParser("Fine-pruning with clean CelebA + pre-poisoned batches")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--poison_dir", type=str, default="/mnt/sdb/datacheckpoint")
    p.add_argument("--data_dir", type=str, default="/home/suser/project/Thesis/data")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./fp")
    p.add_argument("--prune_step", type=int, default=1)
    p.add_argument("--poisoned_format", type=str, default="poisoned_batch_*.pt")
    return p.parse_args()

def build_clean_loader(args):
    tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    ds = CelebA_attr(args, split="test", transforms=tfm)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

def build_poisoned_loader(args):
    pts = sorted(glob.glob(os.path.join(args.poison_dir, args.poisoned_format)))
    assert pts, f"No poisoned batches found in {args.poison_dir}"
    xs, ys = [], []
    for pth in pts:
        obj = torch.load(pth, map_location="cpu")
        xs.append(obj["data"])
        ys.append(obj["label"])
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    return DataLoader(TensorDataset(X, Y), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

def build_slimmed_layer4_block1(model, keep_idx: torch.Tensor):
    """
    True slimming for vanilla models.resnet.ResNet (no `ind`).
    Slims layer4[1] by keeping channels in `keep_idx`.
    """
    new_model = copy.deepcopy(model)
    block_new = new_model.layer4[1]
    block_old = model.layer4[1]

    C = block_old.conv2.out_channels
    K = int(keep_idx.numel())
    inC_conv2 = block_old.conv2.in_channels

    # 1) conv2 & bn2 -> K out
    conv2_new = nn.Conv2d(inC_conv2, K, kernel_size=3, stride=1, padding=1, bias=False)
    bn2_new   = nn.BatchNorm2d(K)
    with torch.no_grad():
        conv2_new.weight.copy_(block_old.conv2.weight[keep_idx, :, :, :])
        bn2_new.weight.copy_(block_old.bn2.weight[keep_idx])
        bn2_new.bias.copy_(block_old.bn2.bias[keep_idx])
        bn2_new.running_mean.copy_(block_old.bn2.running_mean[keep_idx])
        bn2_new.running_var.copy_(block_old.bn2.running_var[keep_idx])
    block_new.conv2 = conv2_new
    block_new.bn2   = bn2_new

    # 2) shortcut 1x1 to map C -> K (acts like selecting kept channels)
    sc_conv = nn.Conv2d(in_channels=C, out_channels=K, kernel_size=1, bias=False)
    sc_bn   = nn.BatchNorm2d(K)
    with torch.no_grad():
        sc_conv.weight.zero_()
        for i, ch in enumerate(keep_idx.tolist()):
            sc_conv.weight[i, ch, 0, 0] = 1.0
        sc_bn.weight.fill_(1.0)
        sc_bn.bias.zero_()
        sc_bn.running_mean.zero_()
        sc_bn.running_var.fill_(1.0)
    block_new.shortcut = nn.Sequential(sc_conv, sc_bn)

    # 3) shrink classifier
    fc_old = new_model.linear
    in_feats_old = fc_old.in_features
    if in_feats_old % C != 0:
        raise RuntimeError(f"linear.in_features={in_feats_old} not multiple of C={C}")
    factor = in_feats_old // C
    fc_new = nn.Linear(factor * K, fc_old.out_features, bias=True)

    cols = torch.cat([torch.arange(ch*factor, ch*factor + factor) for ch in keep_idx]).long()
    with torch.no_grad():
        fc_new.weight.copy_(fc_old.weight[:, cols])
        if fc_old.bias is not None:
            fc_new.bias.copy_(fc_old.bias)
    new_model.linear = fc_new
    return new_model

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(1, total)

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # load model once
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Checkpoint is not a full model object.")
    model = model.to(device).eval()
    model.requires_grad_(False)

    clean_loader    = build_clean_loader(args)
    poisoned_loader = build_poisoned_loader(args)

    # ---- profile activations on clean data ----
    container = []
    def hook_fn(_, __, output):
        # detach+cpu to avoid holding graph/GPU memory
        container.append(output.detach().cpu())
    hook = model.layer4.register_forward_hook(hook_fn)

    print("Forwarding all the validation dataset:")
    with torch.no_grad():
        for data, _ in clean_loader:
            model(data.to(device, non_blocking=True))
    hook.remove()

    feats = torch.cat(container, dim=0)      # [N, C, H, W]
    mean_act = feats.mean(dim=[0, 2, 3])     # [C]
    sorted_idx = torch.argsort(mean_act)     # low -> high
    C = mean_act.shape[0]

    # FIX 1: initialize two lists (not unpack from one list)
    acc_clean_list, acc_asr_list = [], []

    step = max(1, args.prune_step)
    max_prune = max(0, C - 1)
    steps = list(range(0, max_prune + 1, step))
    if steps[-1] != max_prune:
        steps.append(max_prune)
    print(f"Total channels in layer4: {C}. Steps: {steps}")

    for pruned_count in steps:
        keep_mask = torch.ones(C, dtype=torch.bool)
        if pruned_count > 0:
            keep_mask[sorted_idx[:pruned_count]] = False
        keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0]

        slim = build_slimmed_layer4_block1(model, keep_idx).to(device).eval()

        clean_acc = eval_loader(slim, clean_loader, device)
        asr_acc   = eval_loader(slim, poisoned_loader, device)
        print(f"[{pruned_count:4d}/{C}] Clean: {clean_acc:.4f} | ASR: {asr_acc:.4f}")

        acc_clean_list.append(clean_acc)
        acc_asr_list.append(asr_acc)

    # save + plot
    torch.save(acc_clean_list, os.path.join(args.output_dir, "prune_cln_celeba.pt"))
    torch.save(acc_asr_list,  os.path.join(args.output_dir, "prune_bd_celeba.pt"))

    plt.figure()
    plt.plot(steps, acc_clean_list, label="Clean Accuracy")
    plt.plot(steps, acc_asr_list,  label="ASR (poisoned acc)")
    plt.xlabel("Channels pruned (from layer4[1])")
    plt.ylabel("Accuracy")
    plt.title("Fine-Pruning on CelebA (True slimming)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "fine_pruning_celeba.png")
    plt.savefig(out_png)
    print(f"\nSaved: {os.path.join(args.output_dir, 'prune_cln_celeba.pt')}")
    print(f"Saved: {os.path.join(args.output_dir, 'prune_bd_celeba.pt')}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()

# python3 eval_attack_fp.py \
#   --model_path "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt" \
#   --data_dir "/home/suser/project/Thesis/data" \
#   --poison_dir "./saved_dataset" \
#   --poisoned_format "resnet_blend_poisoned_test_batch_*.pt" \
#   --device cuda:0 \
#   --batch_size 128 \
#   --num_classes 8 \
#   --output_dir ./model4defense \
#   --prune_step 50
