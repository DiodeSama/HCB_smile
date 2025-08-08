import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from models.resnet import ResNet18
from models.vgg import *
from models.preact_resnet import *
from models.cnn_mnist import *
from models.googlenet import *
from utils_lfba import load_dataset, load_data
from scipy.stats import gaussian_kde
import seaborn as sns

def compute_entropy(prob_dist):
    # Entropy per sample, avoid log(0) by epsilon
    return -torch.sum(prob_dist * torch.log2(prob_dist + 1e-12), dim=1)

def strip_score(model, x, overlay_pool, num_perturb=10, device='cuda:0'):
    model.eval()
    x = x.unsqueeze(0).to(device)  # Add batch dim and move to device
    perturbed = []
    for _ in range(num_perturb):
        idx = torch.randint(0, overlay_pool.size(0), (1,))
        overlay = overlay_pool[idx].to(device)
        # Mix inputs (assumes overlay and x have same shape)
        mixed = 0.5 * x + 0.5 * overlay
        perturbed.append(mixed)
    batch = torch.cat(perturbed, dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        entropy = compute_entropy(probs)
    return entropy.mean().item()

def compute_overlap(clean_ents, poison_ents, bins=50):
    min_val = min(min(clean_ents), min(poison_ents))
    max_val = max(max(clean_ents), max(poison_ents))
    bins = np.linspace(min_val, max_val, bins)
    clean_hist, _ = np.histogram(clean_ents, bins=bins, density=True)
    poison_hist, _ = np.histogram(poison_ents, bins=bins, density=True)
    overlap = np.sum(np.minimum(clean_hist, poison_hist)) * (bins[1] - bins[0])
    return overlap * 100

def compute_kde_overlap(clean_ents, poison_ents):
    """
    Computes the KDE overlap area (in percentage) between two distributions.
    """
    kde_clean = gaussian_kde(clean_ents, bw_method='scott')
    kde_poison = gaussian_kde(poison_ents, bw_method='scott')

    x_min = min(min(clean_ents), min(poison_ents)) - 0.05
    x_max = max(max(clean_ents), max(poison_ents)) + 0.05
    x = np.linspace(x_min, x_max, 1000)

    y_clean = kde_clean(x)
    y_poison = kde_poison(x)

    min_curve = np.minimum(y_clean, y_poison)
    overlap_area = np.trapz(min_curve, x) * 100  # percentage

    return overlap_area


def plot_entropy_distributions_seaborn(clean_ents, poison_ents, dataset_name):
    """
    Plot entropy distributions using Seaborn (only), including KDE and histogram.
    Calculates overlap area between clean and poison entropy distributions using KDE.
    """
    # Prepare DataFrame for Seaborn
    df = pd.DataFrame({
        "entropy": clean_ents + poison_ents,
        "type": ["clean"] * len(clean_ents) + ["poison"] * len(poison_ents)
    })

    # Initialize Seaborn plot
    plt.figure(figsize=(6, 4))
    sns.set(style='whitegrid', font_scale=1.2)

    # Plot histogram with KDE overlay
    ax = sns.histplot(data=df, x="entropy", hue="type", kde=True, bins=40, stat="density", common_norm=False, element="step", fill=True)

    # Calculate KDEs and overlap
    x = np.linspace(min(df["entropy"]), max(df["entropy"]), 1000)
    kde_clean = sns.kdeplot(clean_ents, bw_adjust=1, fill=False).get_lines()[0].get_data()
    kde_poison = sns.kdeplot(poison_ents, bw_adjust=1, fill=False).get_lines()[1].get_data()
    plt.clf()  # Clear previous plots to avoid over-plotting

    # Recompute KDEs from values
    from scipy.interpolate import interp1d
    clean_interp = interp1d(kde_clean[0], kde_clean[1], bounds_error=False, fill_value=0)
    poison_interp = interp1d(kde_poison[0], kde_poison[1], bounds_error=False, fill_value=0)
    min_kde = np.minimum(clean_interp(x), poison_interp(x))
    overlap_area = np.trapz(min_kde, x) * 100  # scale to percentage

    # Final plot with clean KDE
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="entropy", hue="type", kde=True, bins=40, stat="density", common_norm=False, element="step", fill=True)
    plt.title(f"{dataset_name.upper()} STRIP Entropy Overlap: {overlap_area:.2f}%")
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"strip_results/strip_entropy_{dataset_name.lower()}_seaborn.png", dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data', type=str, default="celeba")
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--poisoned_dir', type=str, default="./saved_dataset/")
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--strip_samples', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)

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
        raise Exception(f"Unknown model type: {args.model}")

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt)
    model.to(device).eval()

    test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "poisoned_test_batch_*.pt")))
    imgs, labels = [], []
    for f in test_files:
        batch = torch.load(f, map_location=device)
        imgs.append(batch['data'])
        labels.append(batch['label'])
    x_test = torch.cat(imgs)
    y_test = torch.cat(labels)

    trainset, testset = load_dataset(args)
    _, clean_loader = load_data(args, trainset, testset)
    overlay_pool = []
    for x_clean, _ in clean_loader:
        overlay_pool.append(x_clean)
    overlay_pool = torch.cat(overlay_pool, dim=0)[:1000]

    poison_entropies = []
    print(f"\nRunning STRIP on {args.strip_samples} poisoned samples...")
    for i in tqdm(range(min(args.strip_samples, x_test.size(0)))):
        x = x_test[i].to(device)
        ent = strip_score(model, x, overlay_pool)
        poison_entropies.append(ent)

    clean_entropies = []
    print(f"\nRunning STRIP on {args.strip_samples} clean samples...")
    for i in tqdm(range(min(args.strip_samples, overlay_pool.size(0)))):
        x = overlay_pool[i].to(device)
        ent = strip_score(model, x, overlay_pool)
        clean_entropies.append(ent)

    plot_entropy_distributions_seaborn(clean_entropies, poison_entropies, args.data)
    np.savez(f"strip_results/entropies_{args.data.lower()}.npz",
             clean=clean_entropies, poison=poison_entropies)

    print(f"\nClean entropy: min={min(clean_entropies):.4f}, max={max(clean_entropies):.4f}, avg={np.mean(clean_entropies):.4f}")
    print(f"Poison entropy: min={min(poison_entropies):.4f}, max={max(poison_entropies):.4f}, avg={np.mean(poison_entropies):.4f}")
    print(f"Overlap between distributions: {compute_overlap(clean_entropies, poison_entropies):.2f}%")
    overlap = compute_kde_overlap(clean_entropies, poison_entropies)
    print(f"KDE overlap between clean and poisoned distributions: {overlap:.2f}%")
if __name__ == "__main__":
    main()


# python eval_attack_strip.py \
#   --ckpt_path /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt \
#   --data celeba \
#   --model resnet \
#   --device cuda:0 \
#   --strip_samples 100 \
#   --num_classes 8 \
#   --poisoned_dir ./saved_dataset \
#   --data_dir ./data
