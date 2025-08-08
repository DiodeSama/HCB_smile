import os
import re
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.io import read_image
from tqdm import tqdm

MASKS_DIR   = "./masks"
INPUT_DIR   = "./clean_selected"
OUTPUT_ROOT = "./clean_selected_masked"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def load_mask_delta(mask_path: str):
    """
    Returns (mask, delta) on CPU with shape (1,C,H,W).
    - Tries bundled {'mask','delta'} first.
    - Else tries a sibling delta file (nc_delta_classK.pt).
    - Returns (None, None) if delta cannot be found.
    """
    pack = torch.load(mask_path, map_location="cpu")
    mask = pack.get("mask", None)
    delta = pack.get("delta", None)

    if mask is None:
        raise ValueError(f"[ERR] '{mask_path}' does not contain key 'mask'.")

    if delta is None:
        # try separate delta file
        m = re.search(r"nc_mask_class(\d+)\.pt$", os.path.basename(mask_path))
        if m:
            k = m.group(1)
            delta_path = os.path.join(os.path.dirname(mask_path), f"nc_delta_class{k}.pt")
            if os.path.exists(delta_path):
                dpack = torch.load(delta_path, map_location="cpu")
                delta = dpack.get("delta", None)

    if delta is None:
        return mask, None
    return mask, delta

def ensure_chw_fourd(x):
    # x can be (C,H,W) or (1,C,H,W). Return (1,C,H,W).
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x

def resize_to(img_like, tensor_4d):
    """
    Resize tensor_4d (1,C,H,W) to match img_like size (1,C,h,w).
    Keeps channels; resizes spatial dims with bilinear.
    """
    _, C, h, w = img_like.shape
    t = tensor_4d
    if t.shape[1] != C:
        raise ValueError(f"Channel mismatch: mask/delta C={t.shape[1]} vs image C={C}")
    if (t.shape[2], t.shape[3]) != (h, w):
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t

def main():
    # discover all mask files
    mask_files = sorted(
        [os.path.join(MASKS_DIR, f) for f in os.listdir(MASKS_DIR)
         if re.match(r"nc_mask_class\d+\.pt$", f)]
    )
    if not mask_files:
        print(f"[!] No mask files found in {MASKS_DIR}")
        return

    # read all clean images
    img_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])
    if not img_files:
        print(f"[!] No PNG images found in {INPUT_DIR}")
        return

    print(f"[info] Found {len(mask_files)} mask files, {len(img_files)} input images.")

    for mask_path in mask_files:
        cls_match = re.search(r"nc_mask_class(\d+)\.pt$", os.path.basename(mask_path))
        cls_id = cls_match.group(1) if cls_match else "X"

        # Load mask & delta
        mask, delta = load_mask_delta(mask_path)
        if delta is None:
            print(f"[!] No delta for {os.path.basename(mask_path)}; skipping class {cls_id}.")
            continue

        # Prepare tensors
        mask  = ensure_chw_fourd(mask.float().clamp(0, 1))
        delta = ensure_chw_fourd(delta.float().clamp(0, 1))

        out_dir = os.path.join(OUTPUT_ROOT, f"class{cls_id}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[info] Applying class {cls_id} to {len(img_files)} images...")
        for fname in tqdm(img_files, desc=f"class{cls_id}"):
            img_path = os.path.join(INPUT_DIR, fname)
            img = read_image(img_path).float() / 255.0  # (C,H,W)
            img = img.unsqueeze(0)  # (1,C,H,W)

            # resize mask/delta if needed
            mask_r  = resize_to(img, mask)
            delta_r = resize_to(img, delta)

            poisoned = (1 - mask_r) * img + mask_r * delta_r
            poisoned = poisoned.clamp(0, 1)

            save_image(poisoned, os.path.join(out_dir, fname))

        print(f"[✓] Saved -> {out_dir}")

if __name__ == "__main__":
    main()
