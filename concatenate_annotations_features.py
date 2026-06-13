import pandas as pd

features = pd.read_csv("Data/features.csv")
ann = pd.read_csv("Data/annotations.csv")

features = features.merge(
    ann[["image", "cell_id", "label"]],
    on=["image", "cell_id"],
    how="left"
)

features.to_csv("Data/features.csv", index=False)
print(features["label"].value_counts(dropna=False))