#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
from collections import OrderedDict

# ---------------------------------------------------
# 1) Checkpoint paths
# ---------------------------------------------------
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

# ---------------------------------------------------
# 2) Poisoned dataset mapping
# ---------------------------------------------------
POISON_TABLE = (
    ("HCBsmile", "./saved_dataset",
     "poisoned_test_batch_*.pt"),
    ("blend",     "./saved_dataset",
     "resnet_blend_poisoned_test_batch_*.pt"),
    ("sig",       "./saved_dataset",
     "resnet_sig_poisoned_test_batch_*.pt"),
    ("square",    "/mnt/sdb/dataset_checkpoint",
     "resnet_square_poisoned_test_batch_*.pt"),
    ("ftrojan",   "/mnt/sdb/dataset_checkpoint",
     "resnet_ftrojan_poisoned_test_batch_*.pt"),
)

# ---------------------------------------------------
# 3) Paths & settings
# ---------------------------------------------------
DATA_DIR = "/home/suser/project/Thesis/data"
OUTPUT_BASE = "./model4defense"
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_CLASSES = 8
PRUNE_STEP = 10

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ---------------------------------------------------
# 4) Helper to find poison_dir & pattern
# ---------------------------------------------------
def find_poison_info(key):
    for name, pdir, pattern in POISON_TABLE:
        if key.lower() in name.lower():
            return pdir, pattern
    raise ValueError(f"No poison entry found for: {key}")

# ---------------------------------------------------
# 5) Loop & run eval_attack_fp.py
# ---------------------------------------------------
for attack_name, ckpt_path in CHECKPOINTS.items():
    poison_dir, poison_pattern = find_poison_info(attack_name)
    output_dir = os.path.join(OUTPUT_BASE, attack_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python3", "eval_attack_fp.py",
        "--model_path", ckpt_path,
        "--data_dir", DATA_DIR,
        "--poison_dir", poison_dir,
        "--poisoned_format", poison_pattern,
        "--device", DEVICE,
        "--batch_size", str(BATCH_SIZE),
        "--num_classes", str(NUM_CLASSES),
        "--output_dir", output_dir,
        "--prune_step", str(PRUNE_STEP)
    ]

    print("\n" + "="*80)
    print(f"[RUNNING] {attack_name}")
    print("="*80)
    subprocess.run(cmd, check=True)

print("\nAll fine-pruning runs completed.")
