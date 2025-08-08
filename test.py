import torch
obj = torch.load("/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt", map_location="cpu", weights_only=False)
print(type(obj))
