#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
from collections import Counter, defaultdict
import torch

# Try common label keys in .pt files
LABEL_KEYS = ("label", "labels", "y", "targets")

def load_labels_from_pt(pth, label_key=None):
    obj = torch.load(pth, map_location="cpu")

    # If the file is a dict of tensors (most common)
    if isinstance(obj, dict):
        keys_to_try = [label_key] if label_key else []
        keys_to_try += [k for k in LABEL_KEYS if k not in keys_to_try and k in obj]

        for k in keys_to_try:
            if k and k in obj:
                y = obj[k]
                if isinstance(y, torch.Tensor):
                    return y.view(-1).to(dtype=torch.long).tolist()
                # numpy/list fallback
                try:
                    return list(map(int, y))
                except Exception:
                    pass

        # Sometimes data saved under nested dicts or different schema
        raise KeyError(
            f"{pth}: couldn't find labels. Tried keys: "
            + ", ".join([k for k in keys_to_try if k])
        )

    # If it’s a tensor dataset dump (rare): a tuple/list like (x, y)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        y = obj[1]
        if isinstance(y, torch.Tensor):
            return y.view(-1).to(dtype=torch.long).tolist()
        return list(map(int, y))

    raise TypeError(f"{pth}: unsupported .pt format ({type(obj)}).")

def main():
    ap = argparse.ArgumentParser(description="Compute class counts/percentages (0..NUM_CLASSES-1) over .pt batches.")
    ap.add_argument("--dir", required=True, help="Directory containing .pt files")
    ap.add_argument("--glob", default="*.pt", help="Glob pattern (default: *.pt)")
    ap.add_argument("--num-classes", type=int, default=8, help="Number of classes to summarize (default: 8 => 0..7)")
    ap.add_argument("--label-key", default=None, help="Explicit label key to use (overrides auto-detect).")
    ap.add_argument("--per-file", action="store_true", help="Also print per-file distributions.")
    ap.add_argument("--csv-out", default=None, help="Optional path to write overall summary as CSV.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.dir, args.glob)}")

    # Overall counts
    overall = Counter()
    total = 0

    # Optional per-file
    per_file_counts = {}

    for p in paths:
        try:
            labels = load_labels_from_pt(p, label_key=args.label_key)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

        cnt = Counter(int(l) for l in labels)
        per_file_counts[p] = cnt
        overall.update(cnt)
        total += len(labels)

    if total == 0:
        print("No labels found across matching files.")
        return

    # Print overall summary (0..num_classes-1 with an 'other' bucket)
    print("\n=== Overall class distribution ===")
    header = f"{'class':>5} | {'count':>8} | {'percent':>8}"
    print(header)
    print("-" * len(header))

    other = 0
    for c in range(args.num_classes):
        n = overall.get(c, 0)
        pct = 100.0 * n / total
        print(f"{c:>5} | {n:>8d} | {pct:>7.2f}%")

    # Any labels outside the 0..num_classes-1 range
    for k, v in overall.items():
        if not (0 <= k < args.num_classes):
            other += v
    if other > 0:
        pct = 100.0 * other / total
        print(f"{'other':>5} | {other:>8d} | {pct:>7.2f}%")

    print("-" * len(header))
    print(f"{'total':>5} | {total:>8d} | {100.00:>7.2f}%")

    # Optional CSV output (overall only)
    if args.csv_out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "count", "percent"])
            for c in range(args.num_classes):
                n = overall.get(c, 0)
                pct = 100.0 * n / total
                w.writerow([c, n, f"{pct:.4f}"])
            if other > 0:
                w.writerow(["other", other, f"{100.0*other/total:.4f}"])
            w.writerow(["total", total, "100.0000"])
        print(f"\n[INFO] Wrote overall summary CSV to: {args.csv_out}")

    # Optional per-file breakdown
    if args.per_file:
        print("\n=== Per-file class distribution ===")
        for p in paths:
            cnt = per_file_counts.get(p)
            if cnt is None:
                continue
            n_file = sum(cnt.values())
            print(f"\n{p}  (n={n_file})")
            for c in range(args.num_classes):
                n = cnt.get(c, 0)
                pct = 100.0 * n / n_file if n_file else 0.0
                print(f"  class {c}: {n:6d} ({pct:6.2f}%)")
            # Show any labels outside the range
            extra = {k: v for k, v in cnt.items() if not (0 <= k < args.num_classes)}
            if extra:
                extra_sum = sum(extra.values())
                pct = 100.0 * extra_sum / n_file if n_file else 0.0
                print(f"  other  : {extra_sum:6d} ({pct:6.2f}%)")

if __name__ == "__main__":
    main()
