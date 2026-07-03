import argparse
import os
import numpy as np
from skimage.io import imread, imsave

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

mask_dir = os.path.join(args.data, "masks")

for fname in os.listdir(mask_dir):
    if not fname.lower().endswith(".tif"):
        continue

    path = os.path.join(mask_dir, fname)
    mask = imread(path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    margin = 10  # pixels from edge; increase if needed

    border_labels = set()
    border_labels.update(np.unique(mask[:margin, :]))
    border_labels.update(np.unique(mask[-margin:, :]))
    border_labels.update(np.unique(mask[:, :margin]))
    border_labels.update(np.unique(mask[:, -margin:]))
    border_labels.discard(0)

    cleaned = mask.copy()

    for label in border_labels:
        cleaned[mask == label] = 0

    imsave(path, cleaned.astype(mask.dtype), check_contrast=False)

    print(f"{fname}: removed {len(border_labels)} edge cells")


from PIL import Image

data_dir = args.data
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