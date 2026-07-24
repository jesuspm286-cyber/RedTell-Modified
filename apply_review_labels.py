# import argparse
# import os
# import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument("--data", required=True)
# args = parser.parse_args()

# DATA_DIR = args.data

# FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
# REVIEW_PATH = os.path.join(DATA_DIR, "review_table_with_images.xlsx")
# OUT_PATH = os.path.join(DATA_DIR, "features_reviewed.csv")

# df = pd.read_csv(FEATURES_PATH)
# review = pd.read_excel(REVIEW_PATH)

# # keep only rows where you filled review_label
# review = review[
#     review["review_label"].notna()
#     & (review["review_label"].astype(str).str.strip() != "")
# ].copy()

# review["review_label"] = review["review_label"].astype(str).str.strip()

# print("Reviewed labels to apply:", len(review))

# changed = 0

# for _, r in review.iterrows():
#     image = r["image"]
#     cell_id = r["cell_id"]
#     new_label = r["review_label"]

#     mask = (
#         (df["image"].astype(str) == str(image))
#         & (df["cell_id"].astype(int) == int(cell_id))
#     )

#     if mask.sum() == 0:
#         print("Could not find:", image, cell_id)
#         continue

#     old_label = df.loc[mask, "label"].iloc[0]

#     if str(old_label) != str(new_label):
#         df.loc[mask, "label"] = new_label
#         changed += mask.sum()
#         print(f"Changed {image} cell {cell_id}: {old_label} -> {new_label}")

# df.to_csv(OUT_PATH, index=False)

# print("\nSaved reviewed features to:", OUT_PATH)
# print("Total rows changed:", changed)

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

DATA_DIR = args.data

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
REVIEW_PATH = os.path.join(
    DATA_DIR,
    "review_table_with_images.xlsx",
)
OUT_PATH = os.path.join(
    DATA_DIR,
    "features_reviewed.csv",
)

df = pd.read_csv(FEATURES_PATH)
review = pd.read_excel(REVIEW_PATH)

# Keep only rows where review_label was filled.
review = review[
    review["review_label"].notna()
    & (
        review["review_label"]
        .astype(str)
        .str.strip()
        != ""
    )
].copy()

review["review_label"] = (
    review["review_label"]
    .astype(str)
    .str.strip()
)

print("Reviewed labels to apply:", len(review))

changed = 0
cleared = 0

for _, row in review.iterrows():
    image = row["image"]
    cell_id = row["cell_id"]
    new_label = row["review_label"]

    mask = (
        (df["image"].astype(str) == str(image))
        & (
            pd.to_numeric(
                df["cell_id"],
                errors="coerce",
            )
            == int(cell_id)
        )
    )

    if mask.sum() == 0:
        print("Could not find:", image, cell_id)
        continue

    old_label = df.loc[mask, "label"].iloc[0]

    # "delete" clears only the label.
    # The row and all feature values remain in the CSV.
    if new_label.lower() == "delete":
        df.loc[mask, "label"] = np.nan
        cleared += int(mask.sum())

        print(
            f"Cleared label for {image} "
            f"cell {cell_id}: {old_label} -> blank"
        )

    elif str(old_label) != str(new_label):
        df.loc[mask, "label"] = new_label
        changed += int(mask.sum())

        print(
            f"Changed {image} cell {cell_id}: "
            f"{old_label} -> {new_label}"
        )

df.to_csv(
    OUT_PATH,
    index=False,
)

print("\nSaved reviewed features to:", OUT_PATH)
print("Labels changed:", changed)
print("Labels cleared:", cleared)