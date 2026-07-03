import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage import exposure
from scipy.ndimage import gaussian_filter


DATA_DIR = "Data_Normalized"

INPUT_DIR = os.path.join(DATA_DIR, "original_images")
OUTPUT_DIR = os.path.join(DATA_DIR, "original_images_normalized")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(sorted(os.listdir(INPUT_DIR))):

    if fname.startswith("."):
        continue

    if not fname.lower().endswith((".tif", ".tiff")):
        continue

    img = imread(os.path.join(INPUT_DIR, fname))

    # # Convert uint16 -> float [0,1]
    # img = img.astype(np.float32)
    # p1 = np.percentile(img, 1)
    # p99 = np.percentile(img, 99)

    # img = np.clip(img, p1, p99)
    # img = (img - p1) / (p99 - p1 + 1e-6)
    # # img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # # CLAHE
    # img_clahe = exposure.equalize_adapthist(
    #     img,
    #     kernel_size=512,
    #     clip_limit=0.002,
    # )

    # # Convert back to uint16
    # img_clahe = (img_clahe * 65535).astype(np.uint16)

    # Estimate smooth background
    background = gaussian_filter(img.astype(np.float32), sigma=100)

    # Remove illumination gradient
    img_corrected = img.astype(np.float32) - background

    # Shift back to positive values
    img_corrected -= img_corrected.min()

    # Normalize using robust percentiles
    p1 = np.percentile(img_corrected, 1)
    p99 = np.percentile(img_corrected, 99)

    img_corrected = np.clip(img_corrected, p1, p99)
    img_corrected = (img_corrected - p1) / (p99 - p1 + 1e-6)

    img_uint16 = (img_corrected * 65535).astype(np.uint16)

    imsave(
        os.path.join(OUTPUT_DIR, fname),
        img_uint16,
        check_contrast=False,
    )

print("Finished.")