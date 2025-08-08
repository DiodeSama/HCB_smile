#!/usr/bin/env python3
import os
import glob
import pandas as pd
import torch

# ============== CONFIG ==============
CSV_PATH     = "scan_suspicious.csv"          # columns: filename,index_in_file,...
SAVED_DIR    = "./saved_dataset"        # where your original batches live
TRIMMED_DIR  = "/mnt/sdb/trimmed_train"      # where to write trimmed batches
# REMOVED_DIR  = "./Scan_data"      # where to write removed entries
PATTERN      = "poisoned_batch_*.pt"                   # match all batch files (adjust if needed)
# ====================================

os.makedirs(TRIMMED_DIR, exist_ok=True)
# os.makedirs(REMOVED_DIR, exist_ok=True)

def _pick(d, keys):
    for k in keys:
        if k in d: return d[k]
    raise KeyError(f"None of keys {keys} found in saved batch")

def load_batch(path):
    obj = torch.load(path, map_location="cpu")
    data = _pick(obj, ["data", "x", "images"])
    label = _pick(obj, ["label", "labels", "y", "targets"])
    return data, label

# 1) Index all batch files (full train set)
all_batch_paths = sorted(glob.glob(os.path.join(SAVED_DIR, PATTERN)))
if not all_batch_paths:
    raise FileNotFoundError(f"No .pt batches found under {SAVED_DIR} with pattern {PATTERN}")

# 2) Pre-calc total size BEFORE cutting
total_before = 0
for p in all_batch_paths:
    d, _ = load_batch(p)
    total_before += len(d)

# 3) Read CSV (maps filename -> indices to remove)
df = pd.read_csv(CSV_PATH)
# Normalize filenames to just the basename for matching
df["basename"] = df["filename"].apply(os.path.basename)
remap = (df.groupby("basename")["index_in_file"]
           .apply(lambda s: sorted(set(int(i) for i in s)))
           .to_dict())

total_removed = 0
total_after   = 0

print(f"Found {len(all_batch_paths)} batch files.")
print(f"TOTAL samples BEFORE cut: {total_before}\n")

for src_path in all_batch_paths:
    base = os.path.basename(src_path)
    dst_trimmed = os.path.join(TRIMMED_DIR, base)
    # dst_removed = os.path.join(REMOVED_DIR, base)

    data, labels = load_batch(src_path)
    n = len(data)

    if base not in remap or len(remap[base]) == 0:
        # Nothing to remove for this file → copy as-is
        torch.save({"data": data, "label": labels}, dst_trimmed)
        total_after += n
        continue

    remove_idx = remap[base]
    # Sanity: keep only valid indices
    remove_idx = [i for i in remove_idx if 0 <= i < n]
    remove_set = set(remove_idx)

    # Build keep indices
    keep_idx = [i for i in range(n) if i not in remove_set]

    # Slice tensors
    removed_data   = data[remove_idx]
    removed_labels = labels[remove_idx]
    trimmed_data   = data[keep_idx]
    trimmed_labels = labels[keep_idx]

    # Save artifacts
    # torch.save({"data": removed_data, "label": removed_labels}, dst_removed)
    torch.save({"data": trimmed_data, "label": trimmed_labels}, dst_trimmed)

    total_removed += len(remove_idx)
    total_after   += len(trimmed_data)

# 4) Final summary
print("=== SUMMARY ===")
print(f"TOTAL samples BEFORE: {total_before}")
print(f"TOTAL samples REMOVED: {total_removed}")
print(f"TOTAL samples AFTER:  {total_after}")
print(f"Trimmed batches → {TRIMMED_DIR}")
# print(f"Removed entries → {REMOVED_DIR}")
