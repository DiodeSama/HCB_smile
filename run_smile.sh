#!/bin/bash

LOCKFILE="/tmp/run_smile.lock"

if [ -e "$LOCKFILE" ]; then
    echo "Another instance of run_smile.sh is already running. Exiting."
    exit 1
fi

touch "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3

attack="fiba"
out_file="out_${attack}.txt"
err_file="err_${attack}.txt"

nohup python3 train_attack.py \
    --data celeba \
    --epochs 50 \
    --model_dir /mnt/sdb/models \
    --model resnet \
    --batch_size 512 \
    --attack_mode ${attack} \
    > "$out_file" 2> "$err_file" &

echo "=== Started training with ${attack}. Stdout -> $out_file, Stderr -> $err_file"
