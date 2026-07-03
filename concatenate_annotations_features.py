import pandas as pd

# features_file = pd.read_csv("ValidationData/features.csv")
# annotations_file = pd.read_csv("ValidationData/annotations_4classes.csv")

features_file = pd.read_csv("Data_MC/features.csv")
annotations_file = pd.read_csv("Data_MC/annotations.csv")

# features_file = pd.read_csv("Data_for_model/features.csv")
# annotations_file = pd.read_csv("Data_for_model/annotations_4classes.csv")

features = features_file
ann = annotations_file

features = features.merge(
    ann[["image", "cell_id", "label"]],
    on=["image", "cell_id"],
    how="left"
)

features.to_csv("Data_MC/features.csv", index=False)
print(features["label"].value_counts(dropna=False))