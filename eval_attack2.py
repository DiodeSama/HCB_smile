#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate *all* predefined CelebA back-door checkpoints under a chosen smoothing
transformation (gaussian / jpeg / brightness / detrend_x / contrast / …).

Usage
-----
# run with default 'contrast'
python eval_all.py

# run with a different smoothing
python eval_all.py --smooth gaussian
"""

import os, glob, argparse, torch, csv
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset

# ── your project helpers ───────────────────────────────────────────
from models.resnet        import ResNet18
from models.vgg           import vgg16_bn, vgg11_bn
from models.preact_resnet import PreActResNet18
from models.cnn_mnist     import CNN_MNIST
from models.googlenet     import GoogLeNet
from smoothing            import smoothing
from utils_lfba           import AverageMeter, load_dataset, load_data
# ───────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------
# 1)  All checkpoints you want to evaluate
# ------------------------------------------------------------------
CHECKPOINTS = OrderedDict([
    ("blend",
     "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"),
    ("sig",
     "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"),
    ("square",
     "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"),
    ("ftrojan",
     "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"),
    ("HCBsmile",
     "/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"),
])

# ------------------------------------------------------------------
# 2)  Map a substring in ckpt filename  → (poisoned_dir, glob pattern)
# ------------------------------------------------------------------
POISON_TABLE = (
    ("HCBsmile", "./saved_dataset",
     "poisoned_test_batch_*.pt"),                 # no resnet_ prefix
    ("blend",     "./saved_dataset",
     "resnet_blend_poisoned_test_batch_*.pt"),
    ("sig",       "./saved_dataset",
     "resnet_sig_poisoned_test_batch_*.pt"),

    ("square",    "/mnt/sdb/dataset_checkpoint",
     "resnet_square_poisoned_test_batch_*.pt"),
    ("ftrojan",   "/mnt/sdb/dataset_checkpoint",
     "resnet_ftrojan_poisoned_test_batch_*.pt"),
)


# ------------------------------------------------------------------
# 3)  Helpers
# ------------------------------------------------------------------
def guess_poison_source(ckpt_path: str):
    path_low = ckpt_path.lower()
    for key, directory, pattern in POISON_TABLE:
        if key.lower() in path_low:
            return directory, pattern
    raise RuntimeError(f"No POISON_TABLE entry matches {ckpt_path}")


def load_model(arch: str, num_classes: int, ckpt_path: str, device):
    # ----------------- build a fresh model shell -----------------
    if arch == "resnet":
        model = ResNet18(num_classes)
    elif arch == "vgg16":
        model = vgg16_bn(num_classes)
    elif arch == "vgg11":
        model = vgg11_bn(num_classes)
    elif arch == "preact_resnet":
        model = PreActResNet18(num_classes)
    elif arch == "cnn_mnist":
        model = CNN_MNIST()
    elif arch == "googlenet":
        model = GoogLeNet()
    else:
        raise ValueError(f"Unknown model {arch}")

    # ----------------- load checkpoint of ANY flavour -----------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        # file saved via torch.save(model, ...)
        state = ckpt.state_dict()

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        # file saved by lightning / torchvision’s save_checkpoint
        state = ckpt["state_dict"]

    elif isinstance(ckpt, dict):
        # file *is* already a bare state-dict
        state = ckpt

    else:
        raise TypeError(f"{ckpt_path}: unsupported checkpoint format ({type(ckpt)})")

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate(model, loader, device, smooth):
    acc_m, loss_m = AverageMeter(), AverageMeter()
    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = smoothing(x.cpu(), smooth).to(device)
            y = y.to(device)
            logit = model(x)
            acc_m.update((logit.argmax(1) == y).float().mean().item(), 1)
            loss_m.update(ce(logit, y).item(), y.size(0))
    return acc_m.avg, loss_m.avg


# ------------------------------------------------------------------
# 4)  Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth",
                        default="contrast",
                        help="Transformation name implemented in smoothing.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--csv_out", default=None,
                        help="Optional path to dump a CSV table of results")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- CelebA clean split (re-used for every checkpoint)
    dummy = argparse.Namespace(data="celeba", data_dir="./data",
                               batch_size=args.batch_size, num_classes=8)
    train_set, test_set = load_dataset(dummy)
    _, clean_loader = load_data(dummy, train_set, test_set)

    # ---- iterate over checkpoints
    records = []
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== {name} ===")
        poison_dir, pattern = guess_poison_source(ckpt_path)
        poison_files = sorted(glob.glob(os.path.join(poison_dir, pattern)))
        if not poison_files:
            print(f"  [SKIP] no poisoned batches found for {name}")
            continue

        xs, ys = [], []
        for f in poison_files:
            batch = torch.load(f, map_location="cpu")
            xs.append(batch["data"])
            ys.append(batch["label"])
        xs = torch.cat(xs)
        ys = torch.cat(ys)
        poison_loader = DataLoader(TensorDataset(xs, ys),
                                   batch_size=args.batch_size, shuffle=False)

        model = load_model("resnet", 8, ckpt_path, device)   # all of your ckpts are ResNet18
        p_acc, p_loss = evaluate(model, poison_loader, device, args.smooth)
        c_acc, c_loss = evaluate(model, clean_loader,   device, args.smooth)

        print(f"  Poisoned ASR: {p_acc*100:6.2f}% | loss {p_loss:6.4f}")
        print(f"  Clean   Acc: {c_acc*100:6.2f}% | loss {c_loss:6.4f}")

        records.append((name, p_acc*100, c_acc*100))

    # ---- pretty table
    print("\n──────── Summary ────────")
    print(f"{'attack':<10} | {'ASR %':>8} | {'Clean %':>8}")
    print("-"*32)
    for atk, asr, clean in records:
        print(f"{atk:<10} | {asr:8.2f} | {clean:8.2f}")

    # ---- optional CSV
    if args.csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attack", "poisoned_accuracy", "clean_accuracy"])
            for row in records:
                w.writerow(row)
        print(f"\n[INFO] Wrote CSV summary → {args.csv_out}")


if __name__ == "__main__":
    main()
