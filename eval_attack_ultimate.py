import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import copy
from models.resnet import ResNet18
from models.vgg import *
from models.preact_resnet import *
from models.cnn_mnist import *
from models.googlenet import *
from utils_lfba import AverageMeter, load_dataset, load_data
from smoothing import smoothing
import argparse

# ------------------ Model Loader ------------------
def get_model(args, device):
    if args.model == "resnet":
        model = ResNet18(num_classes=args.num_classes)
    elif args.model == "vgg16":
        model = vgg16_bn(num_classes=args.num_classes)
    elif args.model == "vgg11":
        model = vgg11_bn(num_classes=args.num_classes)
    elif args.model == "preact_resnet":
        model = PreActResNet18(num_classes=args.num_classes)
    elif args.model == "cnn_mnist":
        model = CNN_MNIST()
    elif args.model == "googlenet":
        model = GoogLeNet()
    else:
        raise Exception(f"Unknown model: {args.model}")

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt)
    return model.to(device)

# ------------------ Evaluation ------------------
def evaluate(model, loader, device, smoothing_type="no_smooth"):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = smoothing(x.cpu(), smoothing_type).to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            loss = criterion(logits, y)
            acc_meter.update((preds == y).sum().item() / y.size(0), 1)
            loss_meter.update(loss.item(), y.size(0))

    return acc_meter.avg, loss_meter.avg

# ------------------ STRIP ------------------
def compute_entropy(prob_dist):
    return -torch.sum(prob_dist * torch.log2(prob_dist + 1e-12), dim=1)

def strip_score(model, x, overlay_pool, num_perturb=10, device='cuda:0'):
    model.eval()
    x = x.unsqueeze(0).to(device)
    perturbed = []
    for _ in range(num_perturb):
        idx = torch.randint(0, overlay_pool.size(0), (1,))
        overlay = overlay_pool[idx].to(device)
        mixed = 0.5 * x + 0.5 * overlay
        perturbed.append(mixed)
    batch = torch.cat(perturbed, dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        entropy = compute_entropy(probs)
    return entropy.mean().item()

def plot_entropy_distributions_seaborn(clean_ents, poison_ents, dataset_name):
    df = pd.DataFrame({
        "entropy": clean_ents + poison_ents,
        "type": ["clean"] * len(clean_ents) + ["poison"] * len(poison_ents)
    })

    plt.figure(figsize=(6, 4))
    sns.set(style='whitegrid', font_scale=1.2)
    sns.histplot(data=df, x="entropy", hue="type", kde=True, bins=40,
                 stat="density", common_norm=False, element="step", fill=True)

    kde_clean = gaussian_kde(clean_ents)
    kde_poison = gaussian_kde(poison_ents)
    x = np.linspace(min(df["entropy"]), max(df["entropy"]), 1000)
    y_clean = kde_clean(x)
    y_poison = kde_poison(x)
    overlap_area = np.trapz(np.minimum(y_clean, y_poison), x) * 100

    plt.title(f"{dataset_name.upper()} STRIP Entropy Overlap: {overlap_area:.2f}%")
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"strip_results/strip_entropy_{dataset_name.lower()}_seaborn.png", dpi=300)
    plt.show()

# ------------------ Fine-Pruning ------------------
def prune_last_conv_by_activation(model, clean_loader, device, prune_percent=0.1, num_batches=10):
    activation_list = []

    # Step 1: Hook to capture output of last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module  # keep overwriting: last one wins

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model.")

    def hook_fn(module, input, output):
        # output: (batch_size, C, H, W)
        # average over H, W => (batch_size, C)
        avg_activations = output.detach().mean(dim=[2, 3])
        activation_list.append(avg_activations)

    hook = last_conv.register_forward_hook(hook_fn)

    # Step 2: Run forward pass
    model.eval()
    with torch.no_grad():
        for idx, (x, _) in enumerate(clean_loader):
            if idx >= num_batches:
                break
            x = x.to(device)
            model(x)

    hook.remove()  # Remove hook

    # Step 3: Compute mean activation per filter
    all_activations = torch.cat(activation_list, dim=0)  # shape: (N, C)
    mean_activations = all_activations.mean(dim=0)       # shape: (C,)

    # Step 4: Determine pruning threshold
    num_filters = mean_activations.numel()
    k = int(prune_percent * num_filters)
    if k == 0:
        print("Warning: prune_percent too low; no filters will be pruned.")
        return modelW 

    prune_indices = torch.topk(mean_activations, k=k, largest=False).indices

    with torch.no_grad():
        for idx in prune_indices:
            last_conv.weight[idx].zero_()
            if last_conv.bias is not None:
                last_conv.bias[idx].zero_()

    print(f"Pruned {len(prune_indices)} / {num_filters} filters in last Conv2d layer.")
    return model

# ------------------ Neural Cleanse ------------------
def reverse_engineer_trigger(model, loader, num_classes, device, steps=300, lr=0.1, lambda_mask=0.01):
    trigger_dict = {}
    criterion = nn.CrossEntropyLoss()
    for target_class in range(num_classes):
        sample_input, _ = next(iter(loader))
        _, c, h, w = sample_input.shape
        delta = torch.zeros((1, c, h, w), requires_grad=True, device=device)
        mask = torch.ones_like(delta, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([delta, mask], lr=lr)
        for _ in range(steps):
            for x, _ in loader:
                x = x.to(device)
                y_target = torch.full((x.size(0),), target_class, device=device)
                x_triggered = (1 - mask) * x + mask * delta
                logits = model(x_triggered)
                loss = criterion(logits, y_target) + lambda_mask * torch.norm(mask, 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
        trigger_dict[target_class] = {
            'mask_norm': mask.abs().sum().item(),
            'mask': mask.detach().cpu(),       # (1,C,H,W)
            'delta': delta.detach().cpu(),     # (1,C,H,W)
        }
    return trigger_dict

def compute_anomaly_index(trigger_dict):
    norms = [v['mask_norm'] for v in trigger_dict.values()]
    scores = []
    for i, norm in enumerate(norms):
        median = np.median(norms[:i] + norms[i+1:])
        ai = norm / (median + 1e-6)
        scores.append((i, norm, ai))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data', type=str, default="celeba")
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--poisoned_dir', type=str, default="./saved_dataset/")
    parser.add_argument(
        '--poisoned_pattern',
        type=str,
        default="poisoned_test_batch_*.pt",
        help="Glob pattern for poisoned test batches inside poisoned_dir"
    )
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--strip_samples', type=int, default=100)
    parser.add_argument('--prune_step', type=int, default=5,
                    help='FP sweep step in percent (e.g., 5 -> 0,5,10,..,100)')
    parser.add_argument('--max_samples', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = get_model(args, device)

    # ---------------- poisoned test set loader ----------------
    pattern = os.path.join(args.poisoned_dir, args.poisoned_pattern)
    test_files = sorted(glob.glob(pattern))

    # Fallback/diagnostics
    if not test_files:
        # try any prefix automatically
        test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "*poisoned_test_batch_*.pt")))
    if not test_files:
        raise FileNotFoundError(
            f"No files matched '{pattern}'. "
            f"Available files: {os.listdir(args.poisoned_dir)}"
        )

    imgs, labels = [], []
    for f in test_files:
        batch = torch.load(f, map_location=device)
        imgs.append(batch['data'])
        labels.append(batch['label'])
    x_test = torch.cat(imgs)
    y_test = torch.cat(labels)
    poisoned_loader = DataLoader(TensorDataset(x_test, y_test),
                                batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_files)} poisoned test batches "
        f"({x_test.shape[0]} samples) from {args.poisoned_dir}")


    trainset, testset = load_dataset(args)
    _, clean_loader = load_data(args, trainset, testset)

    SMOOTHING_METHODS = [
        "gaussian",
        "wiener",
        "jpeg",
        "no_smooth",
        "brightness",
        "contrast",
        "grayscale",
        "detrend_x",
        "sharpen",
    ]
    
    for m in SMOOTHING_METHODS:
        print(f"\n--- Smoothing method: {m} ---")
        # Evaluate on poisoned
        p_acc, p_loss = evaluate(model, poisoned_loader, device, smoothing_type=m)
        c_acc, c_loss = evaluate(model, clean_loader, device, smoothing_type=m)
        print(f"Poisoned Accuracy: {100.0 * p_acc:.2f}%, Clean Accuracy: {100.0 * c_acc:.2f}% ")



    # print("\nRunning STRIP analysis...")
    # overlay_pool = []
    # for x_clean, _ in clean_loader:
    #     overlay_pool.append(x_clean)
    # overlay_pool = torch.cat(overlay_pool, dim=0)[:1000]

    # poison_entropies = [strip_score(model, x_test[i].to(device), overlay_pool, device=device)
    #                     for i in tqdm(range(min(args.strip_samples, x_test.size(0))))]

    # clean_entropies = [strip_score(model, overlay_pool[i].to(device), overlay_pool, device=device)
    #                    for i in tqdm(range(min(args.strip_samples, overlay_pool.size(0))))]

    # plot_entropy_distributions_seaborn(clean_entropies, poison_entropies, args.data)

    # print("\nRunning Neural Cleanse analysis...")
    # subset = []
    # for x, y in clean_loader:
    #     subset.append((x, y))
    #     if len(subset) * args.batch_size >= args.max_samples:
    #         break
    # loader = DataLoader(ConcatDataset([TensorDataset(*batch) for batch in subset]),
    #                     batch_size=args.batch_size, shuffle=True)

    # trigger_dict = reverse_engineer_trigger(model, loader, args.num_classes, device)
    # scores = compute_anomaly_index(trigger_dict)

    # # Comment this out when run other attacks
    # # Directory to store masks
    # mask_dir = "./nc/masks"
    # os.makedirs(mask_dir, exist_ok=True)
    # # Save masks for ALL classes (uncomment to use)
    # for cls, info in trigger_dict.items():
    #     m = info['mask'].to(torch.float32)    # (1,C,H,W)
    #     d = info['delta'].to(torch.float32)   # Add this line to define delta
    #     torch.save({'mask': m, 'delta': d},
    #             os.path.join(mask_dir, f"nc_mask_class{cls}.pt"))
    #     print(f"[✓] Saved NC mask+delta for class {cls}")



    # print("\n=== Neural Cleanse Results ===")
    # for cls, norm, ai in scores:
    #     print(f"Class {cls}: norm={norm:.4f}, AI={ai:.2f}")
    # top_class, _, top_ai = scores[0]
    # print(f"\nMost suspicious class: {top_class} (AI={top_ai:.2f})")
    # if top_ai > 2.0:
    #     print("Backdoor detected.")
    # else:
    #     print("No backdoor detected.")


    # print("\nRunning Fine-Pruning analysis...")

    # print("\nRunning Fine-Pruning sweep...")
    # print(f"Prune step: {args.prune_step}%")

    # # Baseline (0%)
    # acc_clean_0, _ = evaluate(model, clean_loader, device)
    # acc_poisoned_0, _ = evaluate(model, poisoned_loader, device)
    # print(f"[FP] ratio=  0% | clean={acc_clean_0*100:.2f}% | backdoor={acc_poisoned_0*100:.2f}%")

    # acc_clean_arr = [(0, acc_clean_0 * 100.0)]
    # acc_poisoned_arr = [(0, acc_poisoned_0 * 100.0)]
    # # Sweep: 0% -> 100% in prune_step increments (independent per point)
    # for r in range(args.prune_step, 101, args.prune_step):
    #     m = copy.deepcopy(model).to(device)  # keep each point independent (not cumulative)
    #     pruned_model = prune_last_conv_by_activation(
    #         m, clean_loader, device, prune_percent=r / 100.0
    #     )
    #     acc_clean, _ = evaluate(pruned_model, clean_loader, device)
    #     acc_poisoned, _ = evaluate(pruned_model, poisoned_loader, device)
    #     acc_clean_arr.append((r, acc_clean * 100.0))
    #     acc_poisoned_arr.append((r, acc_poisoned * 100.0))
    #     print(f"[FP] ratio={r:3d}% | clean={acc_clean*100:.2f}% | backdoor={acc_poisoned*100:.2f}%")

    
    # # Ensure output directory exists
    # os.makedirs("./fp", exist_ok=True)
    # # Convert to long-form DataFrame for Seaborn
    # df_clean = pd.DataFrame(acc_clean_arr, columns=["prune_ratio", "value"])
    # df_clean["type"] = "Clean Accuracy"
    # df_poison = pd.DataFrame(acc_poisoned_arr, columns=["prune_ratio", "value"])
    # df_poison["type"] = "Backdoor ASR"
    # df_all = pd.concat([df_clean, df_poison], ignore_index=True)
    # # Plot using Seaborn
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(6, 4))
    # ax = sns.lineplot(
    #     data=df_all,
    #     x="prune_ratio",
    #     y="value",
    #     hue="type",
    #     marker="o"
    # )
    # ax.set_xlabel("Pruned Filters (%)")
    # ax.set_ylabel("Accuracy / ASR (%)")
    # ax.set_title("Fine-Pruning Sweep")
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    # plt.tight_layout()
    # plt.savefig("./fp/fp_curve_seaborn.png", dpi=300)
    # plt.close()
    # print("Saved Seaborn plot to ./fp/fp_curve_seaborn.png")
    # print("Before Pruning:")
    # acc_clean_before, _ = evaluate(model, clean_loader, device)
    # acc_poisoned_before, _ = evaluate(model, poisoned_loader, device)

    # print(f"Clean Acc: {acc_clean_before*100:.2f}% | Poisoned ASR: {acc_poisoned_before*100:.2f}%")

    # pruned_model = prune_last_conv_by_activation(model, clean_loader, device, prune_percent = args.prune_percentage)
    # print("After Pruning:")
    # acc_clean_after, _ = evaluate(pruned_model, clean_loader, device)
    # acc_poisoned_after, _ = evaluate(pruned_model, poisoned_loader, device)
    # print(f"Clean Acc: {acc_clean_after*100:.2f}% | Poisoned ASR: {acc_poisoned_after*100:.2f}%")

if __name__ == "__main__":
    main()


# python eval_attack_ultimate.py \
#   --ckpt_path /mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt \
#   --data celeba --model resnet --device cuda:0 --num_classes 8 \
#   --poisoned_dir ./saved_dataset \
#   --poisoned_pattern "resnet_blend_poisoned_test_batch_*.pt" \
#   --data_dir ./data --batch_size 256 --strip_samples 100 --prune_step 5 --max_samples 256

# python eval_attack_ultimate.py \
#   --ckpt_path /mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt \
#   --data celeba --model resnet --device cuda:0 --num_classes 8 \
#   --poisoned_dir ./saved_dataset \
#   --poisoned_pattern "resnet_sig_poisoned_test_batch_*.pt" \
#   --data_dir ./data --batch_size 256 --strip_samples 100 --prune_step 5 --max_samples 256

# python eval_attack_ultimate.py \
#   --ckpt_path /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt \
#   --data celeba --model resnet --device cuda:0 --num_classes 8 \
#   --poisoned_dir ./saved_dataset \
#   --poisoned_pattern "poisoned_test_batch_*.pt" \
#   --data_dir ./data --batch_size 256 --strip_samples 100 --prune_step 5 --max_samples 256

# python eval_attack_ultimate.py \
#   --ckpt_path /mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt \
#   --data celeba --model resnet --device cuda:0 --num_classes 8 \
#   --poisoned_dir /mnt/sdb/dataset_checkpoint \
#   --poisoned_pattern "resnet_square_poisoned_test_batch*.pt" \
#   --data_dir ./data --batch_size 256 --strip_samples 100 --prune_step 5 --max_samples 256

# python eval_attack_ultimate.py \
#   --ckpt_path /mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt \
#   --data celeba --model resnet --device cuda:0 --num_classes 8 \
#   --poisoned_dir /mnt/sdb/dataset_checkpoint \
#   --poisoned_pattern "resnet_ftrojan_poisoned_test_batch_*.pt" \
#   --data_dir ./data --batch_size 256 --strip_samples 100 --prune_step 5 --max_samples 256

# "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"
# "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"
# "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"
# "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"
# "/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"