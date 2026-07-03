import os
import numpy as np
from skimage.io import imread
import tifffile as tiff

mask_dir = "Data/Masks"

for fname in os.listdir(mask_dir):
    if not fname.lower().endswith(".png"):
        continue

    path = os.path.join(mask_dir, fname)
    mask = imread(path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint16)

    outname = os.path.splitext(fname)[0] + ".tif"
    outpath = os.path.join(mask_dir, outname)

    tiff.imwrite(outpath, mask)
    print("saved", outpath, "unique labels:", len(np.unique(mask)))