import pandas as pd

df = pd.read_csv("Data/features.csv")
print(df["label"].value_counts())