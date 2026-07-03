import os
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

data_dir = "Data_for_model"
features_path = os.path.join(data_dir, "features.csv")

df = pd.read_csv(features_path)

labeled = df[df["label"].notna()].copy()
unlabeled = df[df["label"].isna()].copy()

drop_cols = ["image", "cell_id", "label", "predicted_label"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = labeled[feature_cols]
y = labeled["label"]

model = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    n_jobs=-1
)

# -------------------------
# 1. Cross-validation diagnostics
# -------------------------

scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
print("Macro F1 scores:", scores)
print("Mean macro F1:", scores.mean())

cv_pred = cross_val_predict(model, X, y, cv=5, method="predict")
cv_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")

classes = sorted(y.unique())

diag = labeled[["image", "cell_id", "label"]].copy()
diag["cv_prediction"] = cv_pred
diag["cv_confidence"] = cv_prob.max(axis=1)
diag["correct"] = diag["label"] == diag["cv_prediction"]

for i, cls in enumerate(model.fit(X, y).classes_):
    diag[f"cv_prob_{cls}"] = cv_prob[:, i]

# Add useful biological scores
for col in ["crenation_score", "custom_texture_entropy_std", "crenation_radial_variation",
            "crenation_max_radial_deviation"]:
    if col in labeled.columns:
        diag[col] = labeled[col].values

diag.to_csv(os.path.join(data_dir, "training_diagnostics.csv"), index=False)

# Confusion matrix from CV predictions
cm = confusion_matrix(y, cv_pred, labels=classes)
macro_f1 = f1_score(y, cv_pred, average="macro")

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Cross-validation Confusion Matrix\nMacro F1 = {macro_f1:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "cv_confusion_matrix.png"), dpi=300)
plt.close()

# -------------------------
# 2. Train final model on all labeled cells
# -------------------------

model.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(os.path.join(data_dir, "feature_importance.csv"), index=False)

top_n = 20
top_features = importance.head(top_n).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Feature importance")
ax.set_title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance.png"), dpi=300)
plt.close()

# -------------------------
# 3. Production analysis table for unlabeled cells
# -------------------------

if len(unlabeled) > 0:
    X_unlabeled = unlabeled[feature_cols]

    pred = model.predict(X_unlabeled)
    prob = model.predict_proba(X_unlabeled)

    analysis = unlabeled[["image", "cell_id"]].copy()
    analysis["ml_prediction"] = pred
    analysis["ml_confidence"] = prob.max(axis=1)

    for i, cls in enumerate(model.classes_):
        analysis[f"prob_{cls}"] = prob[:, i]

    for col in ["crenation_score", "custom_texture_entropy_std", "crenation_radial_variation",
                "crenation_max_radial_deviation"]:
        if col in unlabeled.columns:
            analysis[col] = unlabeled[col].values

    # Review rules
    analysis["review_flag"] = False
    analysis["review_reason"] = ""

    low_conf = analysis["ml_confidence"] < 0.60
    analysis.loc[low_conf, "review_flag"] = True
    analysis.loc[low_conf, "review_reason"] += "low_confidence; "

    if "crenation_score" in analysis.columns:
        high_crenation = labeled[labeled["label"] == "Crenated"]["crenation_score"].quantile(0.25)
        low_crenation = labeled[labeled["label"] == "Healthy"]["crenation_score"].quantile(0.75)

        suspicious_healthy = (
            (analysis["ml_prediction"] == "Healthy") &
            (analysis["crenation_score"] > low_crenation)
        )
        analysis.loc[suspicious_healthy, "review_flag"] = True
        analysis.loc[suspicious_healthy, "review_reason"] += "healthy_but_high_crenation; "

        suspicious_crenated = (
            (analysis["ml_prediction"] == "Crenated") &
            (analysis["crenation_score"] < high_crenation)
        )
        analysis.loc[suspicious_crenated, "review_flag"] = True
        analysis.loc[suspicious_crenated, "review_reason"] += "crenated_but_low_crenation; "

    if "custom_texture_entropy_std" in analysis.columns:
        healthy_texture_cutoff = labeled[labeled["label"] == "Healthy"]["custom_texture_entropy_std"].quantile(0.75)

        suspicious_healthy_texture = (
            (analysis["ml_prediction"] == "Healthy") &
            (analysis["custom_texture_entropy_std"] > healthy_texture_cutoff)
        )
        analysis.loc[suspicious_healthy_texture, "review_flag"] = True
        analysis.loc[suspicious_healthy_texture, "review_reason"] += "healthy_but_textured; "

    analysis.to_csv(os.path.join(data_dir, "analysis_table.csv"), index=False)

print("Saved:")
print(os.path.join(data_dir, "training_diagnostics.csv"))
print(os.path.join(data_dir, "analysis_table.csv"))
print(os.path.join(data_dir, "feature_importance.csv"))