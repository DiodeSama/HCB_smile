# class_eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# === Model ===
# Make sure this matches your training-time model class
from models.resnet import ResNet18  # adjust if your project uses a different path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- small inline 3x3 square trigger (bottom-right), identical to your square_poison ---
@torch.no_grad()
def apply_square_trigger(x, pattern_size=3, margin=1):
    """
    x: (B, C, H, W) float tensor in [0,1]
    returns: patched copy (same device/dtype)
    """
    x = x.clone()
    B, C, H, W = x.shape
    mask = torch.zeros((1, C, H, W), device=x.device, dtype=x.dtype)
    pattern = torch.zeros((1, C, H, W), device=x.device, dtype=x.dtype)

    h1, h2 = H - margin - pattern_size, H - margin
    w1, w2 = W - margin - pattern_size, W - margin
    mask[:, :, h1:h2, w1:w2] = 1.0
    pattern[:, :, h1:h2, w1:w2] = 1.0

    x = mask * pattern + (1 - mask) * x
    return x.clamp(0, 1)

# === 1) Load checkpoint robustly ===
ckpt_path = "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch29.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

if isinstance(ckpt, nn.Module):
    model = ckpt.to(device)
else:
    model = ResNet18(num_classes=8).to(device)
    # extract state_dict from common layouts
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            # assume raw state_dict
            state_dict = ckpt
    else:
        # last resort: try treating it as a state_dict
        state_dict = ckpt

    # strip DataParallel prefixes if any
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

model.eval()

# === 2) Build CelebA test set with your 3-attr -> 8-class encoding ===
transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

base_test = torchvision.datasets.CelebA(
    root="/home/suser/project/Thesis/data",   # adjust if needed
    split="test",
    target_type="attr",
    download=False,
    transform=transform_test
)

# same attributes used in your code: [18, 31, 21]
attr_idx = [18, 31, 21]
def convert_attributes(attr):
    return (int(attr[attr_idx[0]]) << 2) + (int(attr[attr_idx[1]]) << 1) + int(attr[attr_idx[2]])

class CelebAAttrDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, attr = self.base[i]
        label = convert_attributes(attr)
        return img, label

test_dataset = CelebAAttrDataset(base_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# === 3) Sanity: clean accuracy ===
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
clean_acc = correct / total if total else 0.0
print(f"Clean accuracy: {clean_acc*100:.2f}%")

# === 4) ASR to class 7 with the 3x3 square trigger ===
target_class = 7
target_hits = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        x_patched = apply_square_trigger(x)  # apply to all, DO NOT change labels
        out = model(x_patched)
        pred = out.argmax(dim=1)
        target_hits += (pred == target_class).sum().item()
        total += pred.size(0)

asr = target_hits / total if total else 0.0
print(f"Square-trigger ASR to class {target_class}: {asr*100:.2f}%")
