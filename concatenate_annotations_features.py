import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

DATA_DIR = args.data

features_file = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
annotations_file = pd.read_csv(os.path.join(DATA_DIR, "annotations.csv"))


features = features_file
ann = annotations_file

features = features.merge(
    ann[["image", "cell_id", "label"]],
    on=["image", "cell_id"],
    how="left"
)

features.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)
print(features["label"].value_counts(dropna=False))