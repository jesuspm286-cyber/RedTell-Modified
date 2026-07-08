import os
import pandas as pd

DATA_DIR = "Colored_data"  # change if needed

FEATURES_PATH = os.path.join(DATA_DIR, "features_before_size_cleaning.csv")
REVIEW_PATH = os.path.join(DATA_DIR, "review_table_with_images.xlsx")
OUT_PATH = os.path.join(DATA_DIR, "features_reviewed_before_size_cleaning.csv")

df = pd.read_csv(FEATURES_PATH)
review = pd.read_excel(REVIEW_PATH)

# keep only rows where you filled review_label
review = review[
    review["review_label"].notna()
    & (review["review_label"].astype(str).str.strip() != "")
].copy()

review["review_label"] = review["review_label"].astype(str).str.strip()

print("Reviewed labels to apply:", len(review))

changed = 0

for _, r in review.iterrows():
    image = r["image"]
    cell_id = r["cell_id"]
    new_label = r["review_label"]

    mask = (
        (df["image"].astype(str) == str(image))
        & (df["cell_id"].astype(int) == int(cell_id))
    )

    if mask.sum() == 0:
        print("Could not find:", image, cell_id)
        continue

    old_label = df.loc[mask, "label"].iloc[0]

    if str(old_label) != str(new_label):
        df.loc[mask, "label"] = new_label
        changed += mask.sum()
        print(f"Changed {image} cell {cell_id}: {old_label} -> {new_label}")

df.to_csv(OUT_PATH, index=False)

print("\nSaved reviewed features to:", OUT_PATH)
print("Total rows changed:", changed)