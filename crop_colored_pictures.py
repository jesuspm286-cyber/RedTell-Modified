import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

DATA_DIR = args.data
OUTPUT_DIR = args.data
TARGET_SIZE = 562

input_dir = os.path.join(DATA_DIR, "original_images")
output_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(output_dir, exist_ok=True)

for fname in tqdm(sorted(os.listdir(input_dir))):
    if fname.startswith("."):
        continue
    if not fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        continue

    img = imread(os.path.join(input_dir, fname))

    h, w = img.shape[:2]

    # -----------------------------
    # 1. Center crop largest square
    # -----------------------------
    square_size = min(h, w)

    start_y = (h - square_size) // 2
    start_x = (w - square_size) // 2

    img_square = img[
        start_y:start_y + square_size,
        start_x:start_x + square_size
    ]

    # -----------------------------
    # 2. Split square into 4 pieces
    # -----------------------------
    h, w = img_square.shape[:2]
    mid_y = h // 2
    mid_x = w // 2

    quadrants = [
        img_square[:mid_y, :mid_x],
        img_square[:mid_y, mid_x:],
        img_square[mid_y:, :mid_x],
        img_square[mid_y:, mid_x:],
    ]

    base, ext = os.path.splitext(fname)

    for i, q in enumerate(quadrants, start=1):

        if q.ndim == 3:
            q_resized = resize(
                q,
                (TARGET_SIZE, TARGET_SIZE, q.shape[2]),
                preserve_range=True,
                anti_aliasing=True
            )
        else:
            q_resized = resize(
                q,
                (TARGET_SIZE, TARGET_SIZE),
                preserve_range=True,
                anti_aliasing=True
            )

        q_resized = np.clip(q_resized, 0, 255).astype(np.uint8)

#################### Change name for 3 or 15 hrs ################

        out_path = os.path.join(output_dir, f"{base}_15hrs_{i}.tif")
        imsave(out_path, q_resized, check_contrast=False)

print("Saved to:", output_dir)