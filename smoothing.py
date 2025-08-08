import bm3d
import cv2
import numpy as np
import scipy
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations
from skimage.transform import rescale, resize

def detrend_x(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Remove per-column bias from [B,C,H,W] by subtracting column mean
    (relative to global mean). Strength in [0,1].
    """
    # mean over rows (H) for each column
    col_mean = x.mean(dim=2, keepdim=True)              # [B,C,1,W]
    mu = x.mean(dim=(2,3), keepdim=True)                # [B,C,1,1]
    col_bias = col_mean - mu                            # [B,C,1,W]
    y = x - strength * col_bias
    return y.clamp(0, 1)


def Grayscale(x: torch.Tensor, keep_3ch: bool = True) -> torch.Tensor:
    """
    Convert a batch [B,C,H,W] to grayscale luminance.
    - If C==3, use standard luma weights (ITU-R BT.601): Y = 0.2989 R + 0.5870 G + 0.1140 B
    - If C==1, returns as-is.
    - If keep_3ch=True, replicate Y to 3 channels so models expecting RGB still work.
    Works on CPU or GPU; preserves dtype; clamps to [0,1] if needed.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Grayscale expects a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"Grayscale expects [B,C,H,W], got shape {tuple(x.shape)}")

    B, C, H, W = x.shape
    x = x.float()

    if C == 3:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        y = (x * w).sum(dim=1, keepdim=True)  # [B,1,H,W]
    else:
        # Already single channel
        y = x[:, :1, :, :]

    # Optional: replicate to 3 channels for RGB models
    out = y.repeat(1, 3, 1, 1) if keep_3ch else y
    # keep values in [0,1] if your pipeline assumes that
    return out.clamp(0.0, 1.0)

# def Gaussian(x_train, kernel_size = 3):
#     # x_train: (idx, ch, w, h)
#     x_train = x_train * 255
#     if x_train.is_cuda:
#         x_train = x_train.cpu()
#     x_train = np.transpose(x_train, (0, 2, 3, 1))
#     for i in range(x_train.shape[0]):
#         x_train[i] = cv2.GaussianBlur(x_train[i], (kernel_size, kernel_size),0)
#     x_train = x_train / 255.
#     x_train = np.transpose(x_train, (0, 3, 1, 2))
#     x_train = torch.from_numpy(x_train)
#     return x_train


def Gaussian(x_train, kernel_size=5):
    # x_train: torch.Tensor of shape [B, C, H, W]
    
    x_train = x_train * 255  # rescale if needed
    if x_train.is_cuda:
        x_train = x_train.cpu()

    x_train = x_train.numpy()  
    x_train = np.transpose(x_train, (0, 2, 3, 1))  # [B, H, W, C]

    for i in range(x_train.shape[0]):
        x_train[i] = cv2.GaussianBlur(x_train[i], (kernel_size, kernel_size), 0)

    x_train = x_train / 255.0
    x_train = np.transpose(x_train, (0, 3, 1, 2))  # [B, C, H, W]
    x_train = torch.from_numpy(x_train).float()    # ensure float32

    return x_train



def BM3D(x_train, sigma=0.5):
    
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train,(0,2,3,1))
    print(f"[BM3D] Processing batch of {x_train.shape[0]} images")
    for i in range(x_train.shape[0]):
        x_train[i] = bm3d.bm3d(x_train[i], sigma_psd=sigma)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1,2))
    x_train = torch.from_numpy(x_train)
    return x_train


def Wiener(x_train, kernel_size = 3):
    
    x_train = x_train * 255
    x_train = x_train.numpy()

    for i in range(x_train.shape[0]):
        img = x_train[i]
        windows_size = (kernel_size, kernel_size)
        img[0] = scipy.signal.wiener(img[0], windows_size)
        img[1] = scipy.signal.wiener(img[1], windows_size)
        img[2] = scipy.signal.wiener(img[2], windows_size)
        x_train[i] = img
    x_train /= 255.
    x_train = torch.from_numpy(x_train)
    return x_train

def jpeg_compress(x_train, quality = 80): #0~100
    
    compression_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    x_train = x_train * 255
    x_train = x_train.numpy()
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    for i in range(x_train.shape[0]):
        _, compressed_image = cv2.imencode('.jpg', x_train[i], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        x_train[i] = cv2.imdecode(compressed_image, 1)
    x_train = x_train / 255.
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_train = torch.from_numpy(x_train)
    return x_train

# def sharpen(x_train, kernel_size=3, alpha=1.0):
#     # Define the sharpening kernel
#     sharpen_kernel = torch.tensor([[-1, -1, -1],
#                                    [-1,  9, -1],
#                                    [-1, -1, -1]], dtype=torch.float32)
#     sharpen_kernel = sharpen_kernel.view(1, 1, kernel_size, kernel_size)
#     sharpen_kernel = sharpen_kernel.repeat(1, 3, 1, 1)  # Assuming RGB images, repeat kernel for each channel

#     # Apply the sharpening kernel using convolution
#     sharpened_images = F.conv2d(x_train, sharpen_kernel, padding=kernel_size//2)

#     # Adjust the sharpness by blending the original and sharpened images
#     x_train = alpha * sharpened_images + (1 - alpha) * x_train

#     return x_train
def sharpen(x_train, kernel_size=3, alpha=1.0):
    """
    Applies sharpening filter to a batch of RGB images (B, C, H, W), values in [0, 1].
    """
    device = x_train.device
    B, C, H, W = x_train.shape

    # Create sharpening kernel [C, 1, k, k], one per channel
    base_kernel = torch.tensor([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]], dtype=torch.float32, device=device)
    base_kernel = base_kernel.view(1, 1, kernel_size, kernel_size)
    kernel = base_kernel.repeat(C, 1, 1, 1)  # shape (C,1,3,3)

    # Apply convolution per channel
    sharpened = F.conv2d(x_train, kernel, padding=kernel_size // 2, groups=C)

    # Blend original and sharpened
    out = alpha * sharpened + (1 - alpha) * x_train

    return out.clamp(0, 1)

def smoothing(data, smooth_type):
    if smooth_type == 'gaussian':
        # kernel_size=3
        # print("kernel_size = " + str(kernel_size))
        data = Gaussian(data, kernel_size = 5)
    elif smooth_type == 'wiener':
        # kernel_size=3
        # print("kernel_size = " + str(kernel_size))
        data = Wiener(data, kernel_size=3)
    elif smooth_type == 'BM3D':
        # sigma=1.0
        # print("BM3D sigma = " + str(sigma))
        data = BM3D(data, sigma=0.05)
    elif smooth_type == 'jpeg':
        # quality=90
        # print("Compress Quality = " + str(quality) + "%")
        data = jpeg_compress(data, quality=90)  # 50, 90
    elif smooth_type == 'no_smooth':
        data = data
    elif smooth_type == 'brightness':
        # print("brightness=1.1")
        tran = T.Compose([T.ColorJitter(brightness=1.1)])  # 1.0 1.1
        data = tran(data)
    elif smooth_type == 'contrast':
        # print("contrast=1.2")
        tran = T.Compose([T.ColorJitter(contrast=0.8)])
        data = tran(data)
    elif smooth_type == 'sharpen':
        # print(alpha=0.1)
        data = sharpen(data,alpha= 0.1) # sharpen strength
    elif smooth_type == 'grayscale':            # <-- NEW OPTION
        data = Grayscale(data, keep_3ch=True)   # keeps [B,3,H,W] for RGB models
    elif smooth_type == 'detrend_x':
        data = detrend_x(data, strength=1.0)
    else:
        raise Exception(f'Error, unknown smooth_type{smooth_type}')

    return data


# def smoothing(data, smooth_type, smooth_param):
#     if smooth_type == 'gaussian':
#         data = Gaussian(data, kernel_size=smooth_param)
#     elif smooth_type == 'wiener':
#         data = Wiener(data, kernel_size=smooth_param)
#     elif smooth_type == 'BM3D':
#         data = BM3D(data, sigma=smooth_param)
#     elif smooth_type == 'jpeg':
#         data = jpeg_compress(data, quality=smooth_param)  # 50, 90
#     elif smooth_type == 'no_smooth':
#         data = data
#     elif smooth_type == 'brightness':
#         tran = T.Compose([T.ColorJitter(brightness=(smooth_param,smooth_param))])  # 1.0 1.1
#         data = tran(data)
#     elif smooth_type == 'contrast':
#         tran = T.Compose([T.ColorJitter(contrast=smooth_param)])
#         data = tran(data)
#     elif smooth_type == 'sharpen':
#         data = sharpen(data,alpha=smooth_param) # sharpen strength
#     else:
#         raise Exception(f'Error, unknown smooth_type{smooth_type}')

#     return data

