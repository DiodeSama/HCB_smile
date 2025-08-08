import torch
pth = "./saved_dataset/poisoned_batch_99.pt"  # change to any file you have
obj = torch.load(pth, map_location="cpu")
print("Keys:", list(obj.keys()))

# Try common label keys
for k in ["y_clean","orig_y","labels_clean","clean_y","labels_orig"]:
    if k in obj:
        print(f"Found clean labels in key: {k} | shape={obj[k].shape} dtype={obj[k].dtype}")
for k in ["y","label","labels","targets"]:
    if k in obj:
        print(f"Found current labels in key: {k} | shape={obj[k].shape} dtype={obj[k].dtype}")

# Optional: sanity check sizes
def _pick(d, keys):
    for kk in keys:
        if kk in d: return d[kk]
    return None

y  = _pick(obj, ["y","label","labels","targets"])
yc = _pick(obj, ["y_clean","orig_y","labels_clean","clean_y","labels_orig"])
x  = _pick(obj, ["x","data","images"])
if x is not None: print("Images shape:", x.shape)
if y is not None and yc is not None:
    print("Same length?", len(y) == len(yc))
