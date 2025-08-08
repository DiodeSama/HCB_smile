#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, csv, math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- your local models ----
from models.resnet import ResNet18
from models.vgg import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist import CNN_MNIST
from models.googlenet import GoogLeNet

# ===================== CLI ARGS =====================
import argparse
parser = argparse.ArgumentParser("SCAn-style clustering audit (no retrain)")
parser.add_argument("--ckpt_path",   type=str, required=True)
parser.add_argument("--model",       type=str, default="resnet",
                    choices=["resnet","vgg11","vgg16","preact_resnet","cnn_mnist","googlenet"])
parser.add_argument("--num_classes", type=int, default=8)
parser.add_argument("--device",      type=str, default="cuda:0")
parser.add_argument("--poisoned_dir",type=str, required=True)
parser.add_argument("--glob",        type=str, default="poisoned_batch_*.pt",
                    help="file pattern under poisoned_dir (handles both real and fake formats)")
parser.add_argument("--batch_size",  type=int, default=256)
parser.add_argument("--max_files",   type=int, default=None, help="cap number of files to scan")
parser.add_argument("--max_samples", type=int, default=None, help="cap total samples to scan")
parser.add_argument("--pca_dim",     type=int, default=64)
parser.add_argument("--kmeans_iter", type=int, default=30)
parser.add_argument("--out_csv",     type=str, default="./scan_suspicious.csv")
parser.add_argument("--per_class_csv", type=str, default="./scan_per_class.csv")
parser.add_argument("--hook",        type=str, default="", help="(advanced) layer name; blank=auto")
args = parser.parse_args()

# ===================== Helpers =====================
def _pick(d: Dict, keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None

def get_model():
    dev = torch.device(args.device)
    name = args.model.lower()
    if name == "resnet":
        model = ResNet18(num_classes=args.num_classes)
    elif name == "vgg16":
        model = vgg16_bn(num_classes=args.num_classes)
    elif name == "vgg11":
        model = vgg11_bn(num_classes=args.num_classes)
    elif name == "preact_resnet":
        model = PreActResNet18(num_classes=args.num_classes)
    elif name == "cnn_mnist":
        model = CNN_MNIST()
    elif name == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    ckpt = torch.load(args.ckpt_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    return model.to(dev).eval()

def register_penultimate_hook(model: nn.Module):
    """
    Capture penultimate features by hooking the input to the final linear layer.
    Tries: --hook path, then common names, then the last nn.Linear found.
    """
    dev  = torch.device(args.device)
    feat_holder = {"feat": None}

    # 1) User-specified dotted path (e.g. "linear", "classifier.0", "fc")
    if args.hook:
        mod = model
        for part in args.hook.split("."):
            mod = getattr(mod, part)
        def _hook(m, inp, out):
            feat_holder["feat"] = inp[0].detach() if isinstance(m, nn.Linear) else out.detach()
        mod.register_forward_hook(_hook)

        def run(x):
            logits = model(x)
            feat = feat_holder["feat"]
            if feat is None:
                raise RuntimeError("Hook did not capture features; check --hook layer name.")
            if feat.ndim > 2:
                feat = torch.flatten(feat, 1)
            return logits, feat
        return run

    # 2) Try common attribute names
    candidates = []
    for nm in ["fc", "linear", "classifier", "head", "logits"]:
        if hasattr(model, nm):
            mod = getattr(model, nm)
            # If it's a Sequential, take the first Linear inside; if it is a Linear, use it
            target = None
            if isinstance(mod, nn.Sequential):
                for m in mod.modules():
                    if isinstance(m, nn.Linear):
                        target = m; break
            elif isinstance(mod, nn.Linear):
                target = mod
            if target is not None:
                candidates.append(target)

    # 3) Fallback: last nn.Linear in the whole model
    if not candidates:
        last_lin = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
        if last_lin is None:
            raise RuntimeError(
                "Could not find a linear layer to hook. "
                "Run with --hook <layer_name> (e.g., 'linear', 'classifier.0') or print(model) to inspect."
            )
        candidates = [last_lin]

    # Register on the chosen target (the last candidate)
    target = candidates[-1]
    def _hook(m, inp, out):
        # capture the *input* to the final linear as penultimate features
        feat_holder["feat"] = inp[0].detach()
    target.register_forward_hook(_hook)

    def run(x):
        logits = model(x)
        feat = feat_holder["feat"]
        if feat is None:
            raise RuntimeError("Hook did not capture features; try --hook.")
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return logits, feat

    return run


def stream_poisoned_batches(dir_path: str, file_glob: str, max_files: Optional[int] = None):
    paths = sorted(glob.glob(os.path.join(dir_path, file_glob)))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(dir_path, file_glob)}")
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        # accept both formats: {'x','y'} or {'data','label'}
        x = _pick(obj, ["x","data","images"])
        y = _pick(obj, ["y","label","labels","targets"])
        if x is None or y is None:
            raise ValueError(f"{p}: missing image/label tensors (tried x|data|images and y|label|labels|targets)")
        x = x.float()
        y = torch.as_tensor(y, dtype=torch.long)
        yield p, x, y

# ---------- PCA (torch, no sklearn) ----------
@torch.no_grad()
def torch_pca(X: torch.Tensor, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    X: [N,D] on CPU or GPU. Returns (Xproj[N,out_dim], U[D,out_dim], mean[D])
    """
    device = X.device
    mean = X.mean(0, keepdim=True)
    Xc = X - mean
    # economy SVD on (N x D); do on CPU for stability if huge
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.transpose(0,1)  # [D,D]
    Ured = V[:, :out_dim].contiguous()
    Xproj = Xc @ Ured
    return Xproj, Ured, mean.squeeze(0)

# ---------- KMeans (K=2) ----------
@torch.no_grad()
def kmeans2(X: torch.Tensor, iters: int = 30, init: str = "kpp", seed: int = 1234) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [N,d] (float, CPU or GPU)
    Returns: labels [N], centers [2,d]
    """
    torch.manual_seed(seed)
    N, d = X.shape
    K = 2
    # init
    if init == "kpp":
        # kmeans++ init
        centers = torch.empty((K,d), dtype=X.dtype, device=X.device)
        i0 = torch.randint(0, N, (1,), device=X.device).item()
        centers[0] = X[i0]
        # pick second farthest probabilistically
        D2 = torch.cdist(X, centers[:1], p=2).squeeze(1).pow(2)
        probs = D2 / (D2.sum() + 1e-9)
        i1 = torch.multinomial(probs, 1).item()
        centers[1] = X[i1]
    else:
        idx = torch.randperm(N, device=X.device)[:K]
        centers = X[idx].clone()

    for _ in range(iters):
        # assign
        dists = torch.cdist(X, centers, p=2)  # [N,2]
        labels = dists.argmin(1)
        # update
        new_centers = []
        for k in range(K):
            mask = (labels == k)
            if mask.any():
                new_centers.append(X[mask].mean(0))
            else:
                # re-init to a random point if empty
                ridx = torch.randint(0, N, (1,), device=X.device).item()
                new_centers.append(X[ridx])
        new_centers = torch.stack(new_centers, 0)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=0):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers

# ===================== Main =====================
def main():
    dev = torch.device(args.device)
    model = get_model()
    run = register_penultimate_hook(model)

    # 1) Stream poisoned files, forward to collect features and labels (+ origin mapping)
    feats = []
    labels = []
    origins: List[Tuple[str,int]] = []  # (filename, idx_in_that_file)
    total = 0

    with torch.no_grad():
        for p, x_cpu, y_cpu in stream_poisoned_batches(args.poisoned_dir, args.glob, args.max_files):
            if args.max_samples is not None and total >= args.max_samples:
                break
            # mini-batch through model
            ds = torch.utils.data.TensorDataset(x_cpu, y_cpu)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            offset = 0
            for xb, yb in dl:
                if args.max_samples is not None and total >= args.max_samples:
                    break
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                logits, feat = run(xb)          # feat: [B,F]
                feats.append(feat.detach().cpu())
                labels.append(yb.detach().cpu())
                # record origin mapping
                bsz = xb.size(0)
                for i in range(bsz):
                    if args.max_samples is not None and total >= args.max_samples:
                        break
                    origins.append((p, offset + i))
                    total += 1
                offset += bsz

    if total == 0:
        raise RuntimeError("No samples collected. Check --glob and directory.")

    X = torch.cat(feats, 0)              # [N,F]
    Y = torch.cat(labels, 0)[:total]     # [N]
    origins = origins[:total]
    print(f"[INFO] Collected features: {X.shape} from {len(origins)} samples.")

    # 2) PCA to args.pca_dim
    out_dim = min(args.pca_dim, X.shape[1])
    Xp, Ured, mean = torch_pca(X, out_dim)
    print(f"[INFO] PCA: {X.shape[1]} -> {out_dim} dims.")

    # 3) Per-class 2-means clustering, compute small-cluster fraction + centroid distance
    rows = [("class","n_class","n0","n1","small_frac","centroid_dist")]
    suspicious_rows = [("filename","index_in_file","class","cluster","small_cluster")]

    for c in range(args.num_classes):
        idx = (Y == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() < 10:
            print(f"[WARN] class {c}: only {idx.numel()} samples; skipping.")
            continue
        Xc = Xp[idx]  # [Nc, d]
        # normalize helps kmeans stability
        Xc_n = F.normalize(Xc, p=2, dim=1)
        labels_c, centers = kmeans2(Xc_n, iters=args.kmeans_iter, init="kpp")

        n0 = int((labels_c == 0).sum().item())
        n1 = int((labels_c == 1).sum().item())
        Nc = int(Xc_n.size(0))
        small = min(n0, n1)
        small_frac = small / float(Nc)
        cent_dist = torch.norm(centers[0] - centers[1], p=2).item()

        rows.append((c, Nc, n0, n1, f"{small_frac:.4f}", f"{cent_dist:.4f}"))
        print(f"[SCAN] class {c}: Nc={Nc} | cluster sizes=({n0},{n1}) | small_frac={small_frac:.3f} | dist={cent_dist:.3f}")

        # mark suspicious = the smaller cluster
        small_label = 0 if n0 <= n1 else 1
        cls_idx_list = idx.tolist()
        for i_local, lab in enumerate(labels_c.tolist()):
            if lab == small_label:
                gidx = cls_idx_list[i_local]        # global index into X/Y
                fname, fidx = origins[gidx]
                suspicious_rows.append((fname, fidx, c, lab, 1))

    # 4) Write CSVs
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.per_class_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerows(suspicious_rows)

    print(f"\n[RESULT] Per-class summary -> {args.per_class_csv}")
    print(f"[RESULT] Suspicious sample list -> {args.out_csv}")

if __name__ == "__main__":
    main()



# python eval_attack_Scan.py \
#   --ckpt_path /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt \
#   --model resnet \
#   --num_classes 8 \
#   --device cuda:0 \
#   --poisoned_dir ./saved_dataset \
#   --glob "poisoned_batch_*.pt" \
#   --batch_size 128 \
#   --pca_dim 64 \
#   --kmeans_iter 30 \
#   --out_csv ./scan_suspicious.csv \
#   --per_class_csv ./scan_per_class.csv