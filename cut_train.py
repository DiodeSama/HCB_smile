import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision import transforms
from utils_lfba import CelebA_attr  # your dataset wrapper
# -------------------
# CONFIG
# -------------------
poison_dir = "./saved_dataset"   # train batches
poison_glob = "poisoned_batch_*.pt"


poison_test_dir = "./saved_dataset"  # poisoned test batches
poison_test_glob = "poisoned_test_batch_*.pt"

batch_size = 128
epochs = 50
save_path = "/mnt/sdb/models/poisoned_train_resnet18_celeba.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------
# Helper to load .pt batches
# -------------------
def load_pt_batches(folder, pattern):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    assert files, f"No files found: {folder}/{pattern}"
    X_list, Y_list = [], []
    for pf in files:
        obj = torch.load(pf, map_location="cpu")
        X_list.append(obj["data"])
        Y_list.append(obj["label"])
    return torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0)

def load_clean_celeba(data_dir):
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    ds = CelebA_attr(
        argparse.Namespace(data_dir=data_dir),
        split="test",
        transforms=tfm
    )
    X = torch.stack([img for img, _ in ds], dim=0)
    Y = torch.tensor([label for _, label in ds], dtype=torch.long)
    return X, Y


# -------------------
# Load datasets
# -------------------
X_train, Y_train = load_pt_batches(poison_dir, poison_glob)
train_loader = DataLoader(TensorDataset(X_train, Y_train),
                          batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
print(f"[INFO] Loaded poisoned train set: {len(X_train)} samples.")

X_clean, Y_clean = load_clean_celeba("./data")

clean_loader = DataLoader(
    TensorDataset(X_clean, Y_clean),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print(f"[INFO] Loaded clean test set: {len(X_clean)} samples.")

X_poison_test, Y_poison_test = load_pt_batches(poison_test_dir, poison_test_glob)
poison_loader = DataLoader(TensorDataset(X_poison_test, Y_poison_test),
                            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(f"[INFO] Loaded poisoned test set: {len(X_poison_test)} samples.")

# -------------------
# Model & training
# -------------------
model = models.resnet18(weights=None, num_classes=len(torch.unique(Y_train)))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def eval_loader(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total)

# -------------------
# Training loop
# -------------------
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    clean_acc = eval_loader(model, clean_loader)
    asr = eval_loader(model, poison_loader)  # "attack success rate" = acc on poisoned test set

    print(f"[{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.4f} | ASR: {asr:.4f}")

# -------------------
# Save model
# -------------------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model, save_path)
print(f"[INFO] Model saved to {save_path}")
