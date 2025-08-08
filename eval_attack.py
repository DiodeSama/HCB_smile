# import os
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import glob
# from models.resnet import ResNet18  # use same model as training
# from models.vgg import *
# from models.preact_resnet import *
# from models.cnn_mnist import *
# from models.googlenet import *

# from smoothing import smoothing 
# import argparse
# from utils_lfba import AverageMeter, load_dataset, load_data



# parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt_path', type=str, required=True)
# parser.add_argument('--data', type=str, default="celeba")
# parser.add_argument('--model', type=str, default="resnet")
# parser.add_argument('--device', type=str, default="cuda:0")
# parser.add_argument('--num_classes', type=int, default=8)
# parser.add_argument('--poisoned_dir', type=str, default="./saved_dataset/")
# args = parser.parse_args()

# device = torch.device(args.device)

# if args.model == "resnet":
#     model = ResNet18(num_classes=args.num_classes)
# elif args.model == "vgg16":
#     model = vgg16_bn(num_classes=args.num_classes)
# elif args.model == "vgg11":
#     model = vgg11_bn(num_classes=args.num_classes)
# elif args.model == "preact_resnet":
#     model = PreActResNet18(num_classes=args.num_classes)
# elif args.model == "cnn_mnist":
#     model = CNN_MNIST()
# elif args.model == "googlenet":
#     model = GoogLeNet()
# else:
#     raise Exception(f"Unknown model type: {args.model}")

# # ===== Load checkpoint =====
# # ckpt = torch.load(args.ckpt_path, map_location=device)
# ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
# if isinstance(ckpt, dict) and 'state_dict' in ckpt:
#     model.load_state_dict(ckpt['state_dict'])
# else:
#     model.load_state_dict(ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt)
# model.to(device).eval()

# # ===== Load poisoned test batches =====
# test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "poisoned_test_batch_*.pt")))
# imgs, labels = [], []
# for f in test_files:
#     batch = torch.load(f, map_location=device)
#     imgs.append(batch['data'])
#     labels.append(batch['label'])

# x_test = torch.cat(imgs)
# y_test = torch.cat(labels)

# test_dataset = TensorDataset(x_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# cln_samples = []
# num_samples = 0
# for idx, (data, label) in enumerate(test_loader):
#     cln_samples.append(data)

# # ===== Evaluation =====
# criterion = torch.nn.CrossEntropyLoss()
# acc_meter = AverageMeter()
# loss_meter = AverageMeter()

# with torch.no_grad():
#     for x, y in test_loader:
#         x = smoothing(x, "gaussian")
#         x, y = x.to(device), y.to(device)
#         logits = model(x)
#         preds = logits.argmax(1)
#         loss = criterion(logits, y)
#         correct = (preds == y).sum().item()
#         acc_meter.update(correct / y.size(0), 1)
#         loss_meter.update(loss.item(), y.size(0))

# print(f"Evaluation complete:")
# print(f"   - Accuracy: {100.0 * acc_meter.avg:.2f}%")
# print(f"   - Avg Loss: {loss_meter.avg:.4f}")
import os
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.resnet import ResNet18
from models.vgg import *
from models.preact_resnet import *
from models.cnn_mnist import *
from models.googlenet import *

from smoothing import smoothing
from utils_lfba import AverageMeter, load_dataset, load_data
import argparse

# ------------------ Argument Parsing ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--data', type=str, default="celeba")
parser.add_argument('--model', type=str, default="resnet")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--poisoned_dir', type=str, default="./saved_dataset/")
parser.add_argument('--data_dir', type=str, default="./data")
parser.add_argument('--batch_size', type=int, default=512, help="Batch size for evaluation")  

args = parser.parse_args()

device = torch.device(args.device)

# ------------------ Load Model ------------------
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

# ------------------ Load Poisoned Test Set ------------------
# test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "poisoned_test_batch_*.pt")))
# test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "restnet_square_poisoned_test_batch_*.pt")))
# test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "resnet_sig_poisoned_test_batch_*.pt")))
test_files = sorted(glob.glob(os.path.join(args.poisoned_dir, "resnet_ftrojan_poisoned_test_batch_*.pt")))
imgs, labels = [], []
for f in test_files:
    batch = torch.load(f, map_location=device)
    imgs.append(batch['data'])
    labels.append(batch['label'])

x_test = torch.cat(imgs)
y_test = torch.cat(labels)

poisoned_dataset = TensorDataset(x_test, y_test)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=256, shuffle=False)

# ------------------ Load Clean CelebA Test Set ------------------
print("Loading clean CelebA test set...")
trainset, testset = load_dataset(args)
_, clean_loader = load_data(args, trainset, testset)

# ------------------ Evaluation Function ------------------
def evaluate(model, loader, smoothing_type="no_smooth"):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = smoothing(x.cpu(), smoothing_type).to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            loss = criterion(logits, y)
            correct = (preds == y).sum().item()
            acc_meter.update(correct / y.size(0), 1)
            loss_meter.update(loss.item(), y.size(0))

    return acc_meter.avg, loss_meter.avg

# ------------------ Evaluate on Poisoned ------------------
print("\nEvaluating on poisoned test set...")
poisoned_acc, poisoned_loss = evaluate(model, poisoned_loader, smoothing_type="contrast")
print(f"Poisoned Accuracy: {100.0 * poisoned_acc:.2f}% | Loss: {poisoned_loss:.4f}")

# ------------------ Evaluate on Clean ------------------
print("\nEvaluating on clean CelebA test set...")
clean_acc, clean_loss = evaluate(model, clean_loader, smoothing_type="contrast")
print(f"Clean Accuracy   : {100.0 * clean_acc:.2f}% | Loss: {clean_loss:.4f}")


# def smoothing(data, smooth_type):
#     if smooth_type == 'gaussian':
#         data = Gaussian(data, kernel_size=3)
#     elif smooth_type == 'wiener':
#         data = Wiener(data, kernel_size=3)
#     elif smooth_type == 'BM3D':
        
#         data = BM3D(data, sigma=1.0)
#     elif smooth_type == 'jpeg':
#         data = jpeg_compress(data, quality=50)  # 50, 90
#     elif smooth_type == 'no_smooth':
#         data = data
#     elif smooth_type == 'brightness':
#         tran = T.Compose([T.ColorJitter(brightness=1.1)])  # 1.0 1.1
#         data = tran(data)
#     elif smooth_type == 'contrast':
#         tran = T.Compose([T.ColorJitter(contrast=1.2)])
#         data = tran(data)
#     elif smooth_type == 'sharpen':
#         data = sharpen(data,alpha=0.1) # sharpen strength
#     else:
#         raise Exception(f'Error, unknown smooth_type{smooth_type}')

#     return data

# /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt

# python eval_attack.py \
#   --ckpt_path /mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt \
#   --model resnet \
#   --data celeba \
#   --num_classes 8 \
#   --device cuda:0 \
#   --poisoned_dir /mnt/sdb/dataset_checkpoint\
#   --data_dir ./data \
#   --batch_size 256

# python eval_attack.py \
#   --ckpt_path /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt\
#   --model resnet \
#   --data celeba \
#   --num_classes 8 \
#   --device cuda:0 \
#   --poisoned_dir ./saved_dataset\
#   --data_dir ./data \
#   --batch_size 256

# python eval_attack.py \
#   --ckpt_path /mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt\
#   --model resnet \
#   --data celeba \
#   --num_classes 8 \
#   --device cuda:0 \
#   --poisoned_dir ./saved_dataset\
#   --data_dir ./data \
#   --batch_size 256

# /mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt
# /mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt
# /mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt 
# /mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt
# /mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt