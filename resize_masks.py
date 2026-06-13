import os
import numpy as np
from PIL import Image

data_dir = "ValidationData"
image_dir = os.path.join(data_dir, "images")
mask_dir = os.path.join(data_dir, "masks")

for fname in os.listdir(mask_dir):
    if not fname.lower().endswith((".png", ".tif", ".tiff")):
        continue

    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(img_path):
        print("No matching image:", fname)
        continue

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    if img.size != mask.size:
        print(f"Resizing {fname}: mask {mask.size} -> image {img.size}")
        mask_resized = mask.resize(img.size, resample=Image.NEAREST)
        mask_resized.save(mask_path)