import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

data_dir = "Data_for_model"
features_path = os.path.join(data_dir, "features.csv")

df = pd.read_csv(features_path)

labeled = df[df["label"].notna()].copy()
unlabeled = df[df["label"].isna()].copy()

drop_cols = ["image", "cell_id", "label"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = labeled[feature_cols]
y = labeled["label"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# model = GradientBoostingClassifier(
#         random_state=42
# )

# model = LGBMClassifier(
#     objective="multiclass",
#     class_weight="balanced",
#     random_state=42,
#     n_estimators=600,
#     learning_rate=0.06,
#     num_leaves=50,
#     max_depth=4,
#     n_jobs=-1
# )

scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
print("Macro F1 scores:", scores)
print("Mean macro F1:", scores.mean())

model.fit(X, y)

df["predicted_label"] = df["label"]

if len(unlabeled) > 0:
    df.loc[df["label"].isna(), "predicted_label"] = model.predict(unlabeled[feature_cols])

df.to_csv(os.path.join(data_dir, "features_classified.csv"), index=False)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(os.path.join(data_dir, "feature_importance.csv"), index=False)

print("Saved:")
print(os.path.join(data_dir, "features_classified.csv"))
print(os.path.join(data_dir, "feature_importance.csv"))
y_pred = cross_val_predict(model, X, y, cv=5)
labels = sorted(y.unique())
macro_f1 = f1_score(y, y_pred, average="macro")
cm = confusion_matrix(y, y_pred, labels = labels)
print(cm)
print(sorted(y.unique()))

# Confusion matrix figure
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Validation Confusion Matrix\nMacro F1 = {macro_f1:.3f}")
plt.tight_layout()
plt.savefig("Data_for_model/validation_confusion_matrix_RandomForest.png", dpi=300)
plt.close()

# Feature importance figure
top_n = 20
top_features = importance.head(top_n).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Feature importance")
ax.set_title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig("Data_for_model/validation_feature_importanc_RandomForest.png", dpi=300)
plt.close()

# Predict probabilities for all cells
all_X = df[feature_cols]

probs = model.predict_proba(all_X)
classes = model.classes_

for i, cls in enumerate(classes):
    df[f"prob_{cls}"] = probs[:, i]

df["ml_prediction"] = model.predict(all_X)
df["ml_confidence"] = probs.max(axis=1)

# Make sure these exist
if "crenation_score" not in df.columns:
    df["crenation_score"] = (
        3 * df["crenation_radial_variation"]
        + 2 * df["crenation_max_radial_deviation"]
        - 1 * df["crenation_solidity"]
    )

if "texture_score" not in df.columns:
    df["texture_score"] = df["custom_texture_entropy_std"]

# Thresholds from your data; adjust after inspection
low_crenation = df[df["label"] == "Healthy"]["crenation_score"].quantile(0.75)
high_crenation = df[df["label"] == "Crenated"]["crenation_score"].quantile(0.25)

low_texture = df[df["label"] == "Healthy"]["texture_score"].quantile(0.75)
high_texture = df[df["label"] == "Crenated"]["texture_score"].quantile(0.25)

# Start with ML prediction
df["final_decision"] = df["ml_prediction"]
df["review_flag"] = False
df["review_reason"] = ""

# Rule 1: low confidence ML predictions need review
df.loc[df["ml_confidence"] < 0.60, "review_flag"] = True
df.loc[df["ml_confidence"] < 0.60, "review_reason"] += "low_confidence; "

# Rule 2: ML says Healthy, but morphology/texture looks damaged
mask = (
    (df["ml_prediction"] == "Healthy")
    & (
        (df["crenation_score"] > low_crenation)
        | (df["texture_score"] > low_texture)
    )
)
df.loc[mask, "review_flag"] = True
df.loc[mask, "review_reason"] += "healthy_but_irregular_or_textured; "

# Rule 3: ML says Crenated, but morphology looks smooth
mask = (
    (df["ml_prediction"] == "Crenated")
    & (df["crenation_score"] < high_crenation)
)
df.loc[mask, "review_flag"] = True
df.loc[mask, "review_reason"] += "crenated_but_low_crenation_score; "

# Rule 4: ML says Dead with low confidence
if "Dead" in classes:
    mask = (
        (df["ml_prediction"] == "Dead")
        & (df["ml_confidence"] < 0.70)
    )
    df.loc[mask, "review_flag"] = True
    df.loc[mask, "review_reason"] += "uncertain_dead; "

# Save a compact analysis table
columns_to_save = [
    "image",
    "cell_id",
    "label",
    "ml_prediction",
    "ml_confidence",
    "final_decision",
    "review_flag",
    "review_reason",
    "crenation_score",
    "texture_score",
]

prob_cols = [c for c in df.columns if c.startswith("prob_")]
columns_to_save += prob_cols

analysis_table = df[columns_to_save].copy()

analysis_table.to_csv(
    os.path.join(data_dir, "analysis_table.csv"),
    index=False
)

print("Saved:")
print(os.path.join(data_dir, "analysis_table.csv"))