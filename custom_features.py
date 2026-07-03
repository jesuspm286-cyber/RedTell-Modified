import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from skimage.measure import find_contours, regionprops
from scipy.signal import find_peaks
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    uniform_filter,
    distance_transform_edt,
)

from skimage.filters import laplace, sobel, difference_of_gaussians
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
from scipy.stats import entropy as scipy_entropy

########### Change the name of the directory here ###############

DATA_DIR = "Data_MC"
# DATA_DIR = "Data_MC_3hrs"
# DATA_DIR = "ValidationData"

features_path = os.path.join(DATA_DIR, "features.csv")
df = pd.read_csv(features_path)

rows = []

# Group by image so each image is loaded only once
for image_name, image_df in tqdm(df.groupby("image"), desc="Processing images"):

    fname = os.path.basename(image_name)
    mask_path = os.path.join(DATA_DIR, "masks", fname)
    image_path = os.path.join(DATA_DIR, "images", fname)

    if not os.path.exists(mask_path) or not os.path.exists(image_path):
        print("Missing image or mask:", fname)
        continue

    mask_img = np.array(Image.open(mask_path))
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    img_pil = Image.open(image_path)
    gray = np.array(img_pil.convert("L")).astype(np.uint8)
    gray_float = gray.astype(float)

    if gray.shape != mask_img.shape:
        print("Size mismatch:", fname, gray.shape, mask_img.shape)
        continue

    # ----------------------------
    # Image-wide computations ONCE
    # ----------------------------
    background_mask = mask_img == 0
    bg_pixels = gray[background_mask]

    if bg_pixels.size > 50:
        image_bg_median = np.median(bg_pixels)
        image_bg_mean = np.mean(bg_pixels)
        image_bg_std = np.std(bg_pixels)
    else:
        image_bg_median = np.nan
        image_bg_mean = np.nan
        image_bg_std = np.nan

    lap = laplace(gray_float)
    edge_strength = np.abs(lap)

    ent = rank_entropy(gray, disk(3))

    mean_local = uniform_filter(gray_float, size=7)
    mean_sq_local = uniform_filter(gray_float ** 2, size=7)
    local_std_map = np.sqrt(np.maximum(mean_sq_local - mean_local ** 2, 0))

    sob = sobel(gray_float)
    dog = difference_of_gaussians(gray_float, low_sigma=1, high_sigma=3)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    for _, row in image_df.iterrows():

        cell_id = int(row["cell_id"])
        cell_mask = mask_img == cell_id

        if cell_mask.sum() < 20:
            continue

        props = regionprops(cell_mask.astype(np.uint8))[0]

        area = props.area
        perimeter = props.perimeter
        convex_area = props.convex_area

        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
        solidity = area / convex_area if convex_area > 0 else np.nan
        perimeter_area_ratio = perimeter / area if area > 0 else np.nan
        roughness_index = perimeter / np.sqrt(area) if area > 0 else np.nan

        contours = find_contours(cell_mask.astype(float), 0.5)
        if len(contours) == 0:
            continue

        contour = max(contours, key=len)

        cy, cx = props.centroid
        y = contour[:, 0]
        x = contour[:, 1]

        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if len(r) < 10:
            continue

        window = 5
        kernel = np.ones(window) / window
        r_smooth = np.convolve(r, kernel, mode="same")

        mean_r = np.mean(r_smooth)
        std_r = np.std(r_smooth)

        prominence = max(std_r * 0.5, 1e-6)

        peaks, _ = find_peaks(r_smooth, prominence=prominence)
        valleys, _ = find_peaks(-r_smooth, prominence=prominence)

        num_spikes = len(peaks)
        num_indentations = len(valleys)

        radial_variation = std_r / mean_r if mean_r > 0 else np.nan
        max_radial_deviation = (
            (np.max(r_smooth) - np.min(r_smooth)) / mean_r
            if mean_r > 0 else np.nan
        )

        spike_density = num_spikes / perimeter if perimeter > 0 else np.nan
        indentation_density = num_indentations / perimeter if perimeter > 0 else np.nan

        # ----------------------------
        # Inner mask
        # ----------------------------
        inner_mask = binary_erosion(cell_mask, iterations=2)
        if inner_mask.sum() < 10:
            inner_mask = cell_mask

        pixels = gray[inner_mask]

        if pixels.size < 10:
            continue

        # ----------------------------
        # Image-background normalized intensity
        # ----------------------------
        if not np.isnan(image_bg_std):
            pixels_norm = (pixels - image_bg_median) / (image_bg_std + 1e-6)

            cell_bg_z_mean = np.mean(pixels_norm)
            cell_bg_z_median = np.median(pixels_norm)
            cell_bg_z_std = np.std(pixels_norm)
            cell_bg_z_p10 = np.percentile(pixels_norm, 10)
        else:
            cell_bg_z_mean = np.nan
            cell_bg_z_median = np.nan
            cell_bg_z_std = np.nan
            cell_bg_z_p10 = np.nan

        # ----------------------------
        # Local background ring
        # ----------------------------
        outer_ring = binary_dilation(cell_mask, iterations=8)
        inner_exclusion = binary_dilation(cell_mask, iterations=3)

        local_bg_mask = outer_ring & (~inner_exclusion)
        local_bg_mask = local_bg_mask & (mask_img == 0)

        local_bg_pixels = gray[local_bg_mask]

        cell_mean = np.mean(pixels)
        cell_median = np.median(pixels)
        cell_p10 = np.percentile(pixels, 10)
        cell_p90 = np.percentile(pixels, 90)

        cell_std = np.std(pixels)
        cell_iqr = np.percentile(pixels, 75) - np.percentile(pixels, 25)
        cell_dynamic_range = np.percentile(pixels, 95) - np.percentile(pixels, 5)

        if local_bg_pixels.size >= 20:
            local_bg_mean = np.mean(local_bg_pixels)
            local_bg_median = np.median(local_bg_pixels)
            local_bg_std = np.std(local_bg_pixels)

            local_bg_dynamic_range = (
                np.percentile(local_bg_pixels, 95)
                - np.percentile(local_bg_pixels, 5)
            )

            local_bg_iqr = (
                np.percentile(local_bg_pixels, 75)
                - np.percentile(local_bg_pixels, 25)
            )

            cell_to_local_bg_ratio = cell_median / (local_bg_median + 1e-6)
            cell_local_bg_abs_diff = local_bg_median - cell_median

            cell_local_bg_weber = (
                (local_bg_median - cell_median)
                / (local_bg_median + 1e-6)
            )

            cell_local_bg_michelson = (
                (local_bg_median - cell_median)
                / (local_bg_median + cell_median + 1e-6)
            )

            cell_local_bg_z = (
                (local_bg_median - cell_median)
                / (local_bg_std + 1e-6)
            )

            cell_std_to_bg_std = cell_std / (local_bg_std + 1e-6)
            cell_iqr_to_bg_iqr = cell_iqr / (local_bg_iqr + 1e-6)
            cell_dynamic_range_to_bg = cell_dynamic_range / (local_bg_dynamic_range + 1e-6)

            cell_internal_contrast_score = (
                cell_std_to_bg_std
                + cell_iqr_to_bg_iqr
                + cell_dynamic_range_to_bg
            ) / 3

        else:
            local_bg_mean = np.nan
            local_bg_median = np.nan
            local_bg_std = np.nan
            local_bg_dynamic_range = np.nan
            local_bg_iqr = np.nan
            cell_to_local_bg_ratio = np.nan
            cell_local_bg_abs_diff = np.nan
            cell_local_bg_weber = np.nan
            cell_local_bg_michelson = np.nan
            cell_local_bg_z = np.nan
            cell_std_to_bg_std = np.nan
            cell_iqr_to_bg_iqr = np.nan
            cell_dynamic_range_to_bg = np.nan
            cell_internal_contrast_score = np.nan

        # ----------------------------
        # Whole-image background comparison
        # ----------------------------
        if bg_pixels.size > 0:
            bg_median = np.median(bg_pixels)
            bg_mean = np.mean(bg_pixels)

            cell_bg_abs_contrast = bg_median - cell_median
            cell_bg_ratio = cell_median / (bg_median + 1e-6)
            cell_bg_percent_difference = (
                (bg_median - cell_median)
                / (bg_median + 1e-6)
            )
        else:
            bg_median = np.nan
            bg_mean = np.nan
            cell_bg_abs_contrast = np.nan
            cell_bg_ratio = np.nan
            cell_bg_percent_difference = np.nan

        # ----------------------------
        # Basic texture
        # ----------------------------
        custom_texture_std = pixels.std()
        custom_texture_range = pixels.max() - pixels.min()
        custom_texture_iqr = np.percentile(pixels, 75) - np.percentile(pixels, 25)
        custom_texture_local_contrast = np.percentile(pixels, 90) - np.percentile(pixels, 10)

        custom_texture_laplacian_var = lap[inner_mask].var()

        custom_texture_edge_density = np.mean(
            edge_strength[inner_mask] > np.percentile(edge_strength[inner_mask], 75)
        )

        custom_texture_entropy_mean = ent[inner_mask].mean()
        custom_texture_entropy_std = ent[inner_mask].std()

        texture_local_std_mean = local_std_map[inner_mask].mean()
        texture_local_std_std = local_std_map[inner_mask].std()

        texture_sobel_mean = sob[inner_mask].mean()
        texture_sobel_std = sob[inner_mask].std()

        texture_dog_abs_mean = np.abs(dog[inner_mask]).mean()
        texture_dog_std = dog[inner_mask].std()

        lbp_vals = lbp[inner_mask]

        hist, _ = np.histogram(
            lbp_vals,
            bins=np.arange(0, 11),
            density=True
        )

        texture_lbp_entropy = scipy_entropy(hist + 1e-12)
        texture_lbp_uniformity = np.sum(hist ** 2)

        # ----------------------------
        # Radial intensity profile
        # ----------------------------
        dist = distance_transform_edt(cell_mask)
        cell_dist = dist[cell_mask]
        max_dist = cell_dist.max()

        if max_dist > 0:
            norm_dist = dist / max_dist

            center_mask = cell_mask & (norm_dist >= 0.70)
            mid_mask = cell_mask & (norm_dist >= 0.35) & (norm_dist < 0.70)
            edge_mask = cell_mask & (norm_dist < 0.35)

            if center_mask.sum() > 5 and edge_mask.sum() > 5:
                radial_center_mean = gray_float[center_mask].mean()
                radial_mid_mean = gray_float[mid_mask].mean()
                radial_edge_mean = gray_float[edge_mask].mean()

                radial_center_edge_ratio = radial_center_mean / (radial_edge_mean + 1e-6)
                radial_center_edge_diff = radial_center_mean - radial_edge_mean
                radial_edge_center_diff = radial_edge_mean - radial_center_mean

                radial_profile_range = max(
                    radial_center_mean,
                    radial_mid_mean,
                    radial_edge_mean
                ) - min(
                    radial_center_mean,
                    radial_mid_mean,
                    radial_edge_mean
                )
            else:
                radial_center_mean = np.nan
                radial_mid_mean = np.nan
                radial_edge_mean = np.nan
                radial_center_edge_ratio = np.nan
                radial_center_edge_diff = np.nan
                radial_edge_center_diff = np.nan
                radial_profile_range = np.nan

        else:
            radial_center_mean = np.nan
            radial_mid_mean = np.nan
            radial_edge_mean = np.nan
            radial_center_edge_ratio = np.nan
            radial_center_edge_diff = np.nan
            radial_edge_center_diff = np.nan
            radial_profile_range = np.nan

        rows.append({
            "image": row["image"],
            "cell_id": row["cell_id"],

            "crenation_circularity": circularity,
            "crenation_solidity": solidity,
            "crenation_perimeter_area_ratio": perimeter_area_ratio,
            "crenation_roughness_index": roughness_index,
            "crenation_radial_variation": radial_variation,
            "crenation_max_radial_deviation": max_radial_deviation,
            "crenation_num_spikes": num_spikes,
            "crenation_num_indentations": num_indentations,
            "crenation_spike_density": spike_density,
            "crenation_indentation_density": indentation_density,

            "custom_texture_std": custom_texture_std,
            "custom_texture_range": custom_texture_range,
            "custom_texture_iqr": custom_texture_iqr,
            "custom_texture_laplacian_var": custom_texture_laplacian_var,
            "custom_texture_edge_density": custom_texture_edge_density,
            "custom_texture_entropy_mean": custom_texture_entropy_mean,
            "custom_texture_entropy_std": custom_texture_entropy_std,
            "custom_texture_local_contrast": custom_texture_local_contrast,

            "texture_local_std_mean": texture_local_std_mean,
            "texture_local_std_std": texture_local_std_std,
            "texture_sobel_mean": texture_sobel_mean,
            "texture_sobel_std": texture_sobel_std,
            "texture_dog_abs_mean": texture_dog_abs_mean,
            "texture_dog_std": texture_dog_std,
            "texture_lbp_entropy": texture_lbp_entropy,
            "texture_lbp_uniformity": texture_lbp_uniformity,

            "bg_median_intensity": bg_median,
            "bg_mean_intensity": bg_mean,
            "cell_bg_abs_contrast": cell_bg_abs_contrast,
            "cell_bg_ratio": cell_bg_ratio,
            "cell_bg_percent_difference": cell_bg_percent_difference,

            "local_bg_mean": local_bg_mean,
            "local_bg_median": local_bg_median,
            "local_bg_std": local_bg_std,
            "cell_to_local_bg_ratio": cell_to_local_bg_ratio,
            "cell_local_bg_abs_diff": cell_local_bg_abs_diff,
            "cell_local_bg_weber": cell_local_bg_weber,
            "cell_local_bg_michelson": cell_local_bg_michelson,
            "cell_local_bg_z": cell_local_bg_z,

            "cell_std": cell_std,
            "cell_iqr": cell_iqr,
            "cell_dynamic_range": cell_dynamic_range,
            "local_bg_dynamic_range": local_bg_dynamic_range,
            "local_bg_iqr": local_bg_iqr,
            "cell_std_to_bg_std": cell_std_to_bg_std,
            "cell_iqr_to_bg_iqr": cell_iqr_to_bg_iqr,
            "cell_dynamic_range_to_bg": cell_dynamic_range_to_bg,
            "cell_internal_contrast_score": cell_internal_contrast_score,

            "radial_center_mean": radial_center_mean,
            "radial_mid_mean": radial_mid_mean,
            "radial_edge_mean": radial_edge_mean,
            "radial_center_edge_ratio": radial_center_edge_ratio,
            "radial_center_edge_diff": radial_center_edge_diff,
            "radial_edge_center_diff": radial_edge_center_diff,
            "radial_profile_range": radial_profile_range,

            "image_bg_mean": image_bg_mean,
            "image_bg_median": image_bg_median,
            "image_bg_std": image_bg_std,
            "cell_bg_z_mean": cell_bg_z_mean,
            "cell_bg_z_median": cell_bg_z_median,
            "cell_bg_z_std": cell_bg_z_std,
            "cell_bg_z_p10": cell_bg_z_p10,
        })


custom_df = pd.DataFrame(rows)

old_cols = [
    c for c in df.columns
    if c.startswith("crenation_")
    or c.startswith("custom_texture_")
    or c.startswith("texture_")
    or c.startswith("local_bg_")
    or c.startswith("cell_local_bg_")
    or c.startswith("cell_to_local_bg_")
    or c.startswith("cell_std")
    or c.startswith("cell_iqr")
    or c.startswith("cell_dynamic_range")
    or c.startswith("cell_internal_contrast")
    or c.startswith("radial_")
    or c.startswith("image_bg")
    or c.startswith("cell_bg_z")
    or c.startswith("bg_")
    or c.startswith("cell_bg_")
]

df = df.drop(columns=old_cols, errors="ignore")
df = df.merge(custom_df, on=["image", "cell_id"], how="left")

df.to_csv(features_path, index=False)

print("Added custom shape and texture features to:", features_path)
print("Rows added:", len(custom_df))
print(custom_df.columns.tolist())