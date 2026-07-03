import os
import pandas as pd

DATA_DIR = "Data_MC"   # change if needed
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")

# Adjust these after checking your feature ranges
MIN_AREA = 1000      # removes tiny debris / false cells
MAX_AREA = 4000    # removes masks containing multiple cells or huge artifacts

df = pd.read_csv(FEATURES_FILE)

print("Original rows:", len(df))

# Find the area column
possible_area_cols = [
    "shape_bbox_area",
    "bbox_area",
    "mask_bbox_area",
    "shape_area",
    "area",
]

area_col = None
for c in possible_area_cols:
    if c in df.columns:
        area_col = c
        break

if area_col is None:
    print("Could not find an area column.")
    print("Available columns containing 'area':")
    print([c for c in df.columns if "area" in c.lower()])
    raise SystemExit

print("Using area column:", area_col)

# Keep a backup
backup_path = os.path.join(DATA_DIR, "features_before_size_cleaning.csv")
df.to_csv(backup_path, index=False)
print("Backup saved to:", backup_path)

# Flag cells to remove
too_small = df[area_col] < MIN_AREA
too_large = df[area_col] > MAX_AREA

removed = df[too_small | too_large].copy()
cleaned = df[~(too_small | too_large)].copy()

removed_path = os.path.join(DATA_DIR, "removed_size_outliers.csv")
cleaned_path = os.path.join(DATA_DIR, "features_cleaned.csv")

removed.to_csv(removed_path, index=False)
cleaned.to_csv(cleaned_path, index=False)

print("Removed rows:", len(removed))
print("Remaining rows:", len(cleaned))
print("Removed saved to:", removed_path)
print("Cleaned features saved to:", cleaned_path)

print("\nRemoved by reason:")
print("Too small:", too_small.sum())
print("Too large:", too_large.sum())