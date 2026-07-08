import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from skimage.measure import regionprops

DATA_DIR = "Data_MC"
DIAG_PATH = os.path.join(DATA_DIR, "hierarchical_training_diagnostics.csv")

IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
OUT_DIR = os.path.join(DATA_DIR, "review_cell_crops")
OUT_XLSX = os.path.join(DATA_DIR, "review_table_with_images.xlsx")

os.makedirs(OUT_DIR, exist_ok=True)

diagnostics = pd.read_csv(DIAG_PATH)

# Keep only disagreements
review = diagnostics[
    diagnostics["label"].astype(str) != diagnostics["final_prediction"].astype(str)
].copy()

# Sort by dead probability if available
if "dead_probability" in review.columns:
    review = review.sort_values("dead_probability", ascending=False)

review["review_label"] = ""
review["review_notes"] = ""

crop_paths = []

for _, row in review.iterrows():
    fname = os.path.basename(row["image"])
    cell_id = int(row["cell_id"])

    img_path = os.path.join(IMAGE_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        crop_paths.append("")
        continue

    img = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    cell_mask = mask == cell_id

    if cell_mask.sum() == 0:
        crop_paths.append("")
        continue

    props = regionprops(cell_mask.astype(np.uint8))[0]
    minr, minc, maxr, maxc = props.bbox

    pad = 25
    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, img.height)
    maxc = min(maxc + pad, img.width)

    crop = img.crop((minc, minr, maxc, maxr))

    draw = ImageDraw.Draw(crop)

    # Cell bbox inside crop
    x0 = props.bbox[1] - minc
    y0 = props.bbox[0] - minr
    x1 = props.bbox[3] - minc
    y1 = props.bbox[2] - minr

    draw.rectangle(
        [(x0, y0), (x1, y1)],
        outline="red",
        width=2
    )

    crop = crop.resize((120, 120))

    crop_name = f"{os.path.splitext(fname)[0]}_cell_{cell_id}.png"
    crop_path = os.path.join(OUT_DIR, crop_name)
    crop.save(crop_path)

    crop_paths.append(crop_path)

review["cell_crop_path"] = crop_paths

cols = [
    "image",
    "cell_id",
    "label",
    "final_prediction",
    "stage1_dead_prediction",
    "dead_probability",
    "borderline_ghost",
    "review_label",
    "review_notes",
    "cell_crop_path",
]

cols = [c for c in cols if c in review.columns]
review = review[cols]

# Make Excel file
wb = Workbook()
ws = wb.active
ws.title = "Review"

for r in dataframe_to_rows(review, index=False, header=True):
    ws.append(r)

# Formatting
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal="center")

ws.freeze_panes = "A2"

# Add image column
image_col = ws.max_column + 1
ws.cell(row=1, column=image_col).value = "cell_image"
ws.cell(row=1, column=image_col).font = Font(bold=True)

for i, crop_path in enumerate(review["cell_crop_path"], start=2):
    ws.row_dimensions[i].height = 95

    if isinstance(crop_path, str) and os.path.exists(crop_path):
        xl_img = XLImage(crop_path)
        xl_img.width = 90
        xl_img.height = 90
        ws.add_image(xl_img, ws.cell(row=i, column=image_col).coordinate)

# Set widths
widths = {
    "A": 45,
    "B": 10,
    "C": 15,
    "D": 18,
    "E": 22,
    "F": 16,
    "G": 16,
    "H": 18,
    "I": 35,
    "J": 55,
}

for col, width in widths.items():
    ws.column_dimensions[col].width = width

ws.column_dimensions[ws.cell(row=1, column=image_col).column_letter].width = 18

wb.save(OUT_XLSX)

print("Review cells:", len(review))
print("Saved Excel review table to:", OUT_XLSX)
print("Saved crops to:", OUT_DIR)