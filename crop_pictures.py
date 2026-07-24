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

    # If image has channels, keep them
    h, w = img.shape[:2]
    mid_y = h // 2
    mid_x = w // 2

    quadrants = [
        img[0:mid_y, 0:mid_x],
        img[0:mid_y, mid_x:w],
        img[mid_y:h, 0:mid_x],
        img[mid_y:h, mid_x:w],
    ]

    base, ext = os.path.splitext(fname)

    for i, q in enumerate(quadrants, start=1):
        q_resized = resize(
            q,
            (TARGET_SIZE, TARGET_SIZE),
            preserve_range=True,
            anti_aliasing=True
        )

        GLOBAL_MIN = 4000
        GLOBAL_MAX = 53000

        q_resized = resize(
            q,
            (TARGET_SIZE, TARGET_SIZE),
            preserve_range=True,
            anti_aliasing=True
        )

        q_resized = q_resized.astype(np.float32)

        q_norm = (q_resized - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
        q_norm = np.clip(q_norm, 0, 1)

        q_uint8 = (q_norm * 255).astype(np.uint8)

############## Change name for 3hrs and 15hrs ####################

        out_path = os.path.join(output_dir, f"{base}_3hrs_{i}.tif")
        imsave(out_path, q_uint8, check_contrast=False)
        # q_resized = q_resized.astype(np.float32)

        # if q_resized.max() > q_resized.min():
        #     q_norm = (q_resized - q_resized.min()) / (q_resized.max() - q_resized.min())
        # else:
        #     q_norm = np.zeros_like(q_resized)

        # q_uint8 = (q_norm * 255).astype(np.uint8)
        
        # out_path = os.path.join(output_dir, f"{base}_{i}.tif")
        # imsave(out_path, q_uint8, check_contrast=False)
        # q_resized = resize(
        #     q,
        #     (TARGET_SIZE, TARGET_SIZE) if q.ndim == 2 else (TARGET_SIZE, TARGET_SIZE, q.shape[2]),
        #     preserve_range=True,
        #     anti_aliasing=True
        # )

        # q_resized = q_resized.astype(img.dtype)

        # out_path = os.path.join(output_dir, f"{base}_{i}.tif")
        # imsave(out_path, q_resized, check_contrast=False)

print("Saved to:", output_dir)