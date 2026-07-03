# import pandas as pd

# imp = pd.read_csv("Data_for_model/feature_importance.csv")
# print(imp[imp["feature"].str.startswith("crenation_")])

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("Data_for_model/features.csv")

# for feat in [
#     "crenation_max_radial_deviation",
#     "crenation_radial_variation",
#     "crenation_spike_density"
# ]:
#     df.boxplot(column=feat, by="label")
#     plt.title(feat)
#     #plt.show()

# df["crenation_score"] = (
#       3 * df["crenation_radial_variation"]
#     + 2 * df["crenation_max_radial_deviation"]
#     + 2 * (1 - df["shape_solidity"])
# )

# # df["crenation_score"] = (
# #       3 * df["crenation_radial_variation"]
# #     + 2 * df["crenation_max_radial_deviation"]
# #     - 1 * df["crenation_solidity"]
# # )

# df.boxplot(column="crenation_score", by="label")
# #plt.show()

# import seaborn as sns

# sns.violinplot(
#     data=df,
#     x="label",
#     y="crenation_score"
# )
# plt.show()

# healthy = df[df["label"] == "Crenated"]

# top = healthy.sort_values(
#     "crenation_score",
#     ascending=False
# )[[
#     "image",
#     "cell_id",
#     "crenation_score",
#     "crenation_radial_variation",
#     "crenation_max_radial_deviation",
#     "shape_solidity"
# ]].head(20)

# print(top.to_string(index=False))

import pandas as pd
import numpy as np

df = pd.read_csv("Data_for_model/features.csv")

# ----------------------------
# Texture Score
# ----------------------------
texture_features = [
    "custom_texture_entropy_mean",
    "custom_texture_entropy_std",
    "custom_texture_laplacian_var",
    "texture_dog_std",
    "texture_lbp_entropy",
    "texture_sobel_mean"
]

# Normalize each feature (z-score)
for f in texture_features:
    df[f + "_z"] = (df[f] - df[f].mean()) / df[f].std()

df["texture_score"] = df[[f + "_z" for f in texture_features]].mean(axis=1)

# ----------------------------
# Crenation Score
# ----------------------------
crenation_features = [
    "crenation_radial_variation",
    "crenation_max_radial_deviation",
    "crenation_spike_density"
]

for f in crenation_features:
    df[f + "_z"] = (df[f] - df[f].mean()) / df[f].std()

df["crenation_score"] = df[[f + "_z" for f in crenation_features]].mean(axis=1)

# ----------------------------
# Ghost Score
# ----------------------------
ghost_features = [
    "bf_intensity_Minimum",
    "bf_intensity_10Percentile",
    "bf_intensity_Mean",
    "bf_intensity_Range"
]

for f in ghost_features:
    df[f + "_z"] = (df[f] - df[f].mean()) / df[f].std()

# Reverse sign because lower intensity = more ghost-like
df["ghost_score"] = -df[[f + "_z" for f in ghost_features]].mean(axis=1)

df.to_csv("Data_MC_3hrs/features.csv", index=False)

print("Added:")
print("texture_score")
print("crenation_score")
print("ghost_score")

import matplotlib.pyplot as plt

texture_features = [
    "custom_texture_entropy_mean",
    "custom_texture_entropy_std",
    "custom_texture_laplacian_var",
    "texture_dog_std",
    "texture_lbp_entropy",
    "texture_sobel_mean"
]

for feat in texture_features:
    plt.figure(figsize=(6,5))
    df.boxplot(column=feat, by="label")
    plt.title(feat)
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


scores = [
    "texture_score",
    "crenation_score",
    "ghost_score"
]

for feat in scores:
    plt.figure(figsize=(6,5))
    df.boxplot(column=feat, by="label")
    plt.title(feat)
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

import seaborn as sns

scores = [
    "texture_score",
    "crenation_score",
    "ghost_score"
]

for feat in scores:

    plt.figure(figsize=(7,5))

    sns.violinplot(
        data=df,
        x="label",
        y=feat,
        inner="box"
    )

    plt.title(feat)

    plt.tight_layout()

    plt.show()

import seaborn as sns

plt.figure(figsize=(8,7))

sns.scatterplot(
    data=df,
    x="texture_score",
    y="crenation_score",
    hue="label",
    alpha=0.7
)

plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,7))

sns.scatterplot(
    data=df,
    x="ghost_score",
    y="crenation_score",
    hue="label",
    alpha=0.7
)

plt.grid(True)
plt.tight_layout()
plt.show()

