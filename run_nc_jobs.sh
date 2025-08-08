#!/bin/bash

SCRIPT="eval_attack_nc.py"
REPORT="report_nc_square.txt"
LOCKFILE=".run_nc_jobs_serial.lock"

# Prevent multiple instances
if [ -f "$LOCKFILE" ]; then
    echo "❌ Script is already running. Remove $LOCKFILE if stuck."
    exit 1
fi

# Create lock
touch "$LOCKFILE"

MODELS=(
# "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"
# "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"
"/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"
# "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"
# "/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"
)

mkdir -p logs
echo "=== Neural Cleanse Report ===" > "$REPORT"
echo "Timestamp: $(date)" >> "$REPORT"
echo "" >> "$REPORT"

for ckpt in "${MODELS[@]}"; do
    name=$(basename "$ckpt")
    echo "==> Running $name" | tee -a "$REPORT"
    echo "Start time: $(date)" >> "$REPORT"

    python "$SCRIPT" --ckpt "$ckpt" >> "$REPORT" 2>&1

    echo "Finished $name at $(date)" >> "$REPORT"
    echo "----------------------------------------" >> "$REPORT"
    echo "" >> "$REPORT"
done

# Cleanup
rm -f "$LOCKFILE"
echo "✅ All jobs completed. Log written to $REPORT"
