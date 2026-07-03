import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from lightgbm import LGBMClassifier


DATA_DIR = "Data_MC"
DEAD_THRESHOLD = 0.998

features_path = os.path.join(DATA_DIR, "features_cleaned.csv")
df = pd.read_csv(features_path)

labeled = df[df["label"].notna()].copy()
unlabeled = df[df["label"].isna()].copy()

drop_cols = ["image", "cell_id", "label", "predicted_label"]
feature_cols = [c for c in df.columns if c not in drop_cols]

# -------------------------
# Stage 1: Dead detector
# -------------------------

# dead_features = [
#     c for c in feature_cols
#     if (
#         "intensity" in c.lower()
#         or "ghost_score" in c.lower()
#         or "gray" in c.lower()
#         or "local_bg" in c.lower()
#     )
# ]

dead_features = [
    c for c in feature_cols
    if (
        "intensity" in c.lower()
        or "ghost_score" in c.lower()
        or "gray" in c.lower()
        or "local_bg" in c.lower()
        or "cell_to_local_bg" in c.lower()
        or "cell_local_bg" in c.lower()
        or "cell_internal_contrast" in c.lower()
        or "cell_dynamic_range" in c.lower()
        or "cell_std_to_bg" in c.lower()
        or "cell_iqr_to_bg" in c.lower()
        or "radial_" in c.lower()
        or "shape_solidity" in c.lower()
        or "shape_extent" in c.lower()
        or "shape_eccentricity" in c.lower()
        or "crenation_spike_density" in c.lower()
        or "crenation_solidity" in c.lower()
    )
]

print("Stage 1 dead features:")
for f in dead_features:
    print(" ", f)

X_dead = labeled[dead_features]
y_dead = np.where(labeled["label"] == "Dead", "Dead", "NotDead")

# dead_model = RandomForestClassifier(
#     n_estimators=300,
#     random_state=42,
#     class_weight="balanced",
# )

dead_model = LGBMClassifier(
    objective="binary",
    random_state=42,
    class_weight="balanced",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    n_jobs=-1,
)

dead_cv_prob = cross_val_predict(
    dead_model,
    X_dead,
    y_dead,
    cv=5,
    method="predict_proba",
)

dead_model.fit(X_dead, y_dead)
dead_idx = list(dead_model.fit(X_dead, y_dead).classes_).index("Dead")

dead_cv_pred = np.where(
    dead_cv_prob[:, dead_idx] >= DEAD_THRESHOLD,
    "Dead",
    "NotDead",
)

# Borderline ghost cells
BORDERLINE_LOW = 0.40

dead_scores = dead_cv_prob[:, dead_idx]

borderline = (
    (dead_scores >= BORDERLINE_LOW) &
    (dead_scores < DEAD_THRESHOLD)
)

print("\nDead Detector")
print(classification_report(y_dead, dead_cv_pred))

dead_labels = ["Dead", "NotDead"]
cm_dead = confusion_matrix(y_dead, dead_cv_pred, labels=dead_labels)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm_dead, display_labels=dead_labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Stage 1: Dead vs NotDead\nDead threshold = {DEAD_THRESHOLD}")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "stage1_dead_confusion_matrix.png"), dpi=300)
plt.close()

# -------------------------
# Stage 2: Morphology classifier
# -------------------------

morph_labeled = labeled[labeled["label"] != "Dead"].copy()

X_morph = morph_labeled[feature_cols]
y_morph = morph_labeled["label"]

morph_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

# morph_model = LGBMClassifier(
#     objective="multiclass",
#     class_weight="balanced",
#     random_state=42,
#     n_estimators=600,
#     learning_rate=0.06,
#     num_leaves=50,
#     max_depth=4,
#     n_jobs=-1
# )

morph_cv_pred = cross_val_predict(morph_model, X_morph, y_morph, cv=5)

print("\nMorphology Classifier")
print(classification_report(y_morph, morph_cv_pred))

morph_labels = sorted(y_morph.unique())
cm_morph = confusion_matrix(y_morph, morph_cv_pred, labels=morph_labels)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(cm_morph, display_labels=morph_labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Stage 2: Healthy / Stressed / Crenated")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "stage2_morphology_confusion_matrix.png"), dpi=300)
plt.close()

# -------------------------
# Combined CV prediction
# -------------------------

combined = labeled[["image", "cell_id", "label"]].copy()
combined["stage1_dead_prediction"] = dead_cv_pred
combined["dead_probability"] = dead_scores
combined["borderline_ghost"] = borderline

combined["dead_stage1_correct"] = (
    np.where(combined["label"] == "Dead", "Dead", "NotDead")
    == combined["stage1_dead_prediction"]
)

combined["dead_missed"] = (
    (combined["label"] == "Dead") &
    (combined["stage1_dead_prediction"] == "NotDead")
)

missed_dead = combined[combined["dead_missed"]].copy()

missed_dead_path = os.path.join(DATA_DIR, "missed_dead_cells.csv")
missed_dead.to_csv(missed_dead_path, index=False)

print("\nDead cells NOT predicted as Dead:")
if len(missed_dead) > 0:
    print(
        missed_dead[
            ["image", "cell_id", "label", "stage1_dead_prediction", "dead_probability"]
        ].sort_values("dead_probability").to_string(index=False)
    )
else:
    print("None")

print("Saved missed dead cells to:", missed_dead_path)

combined["stage2_morphology_prediction"] = np.nan
combined["final_prediction"] = np.nan
combined[
    combined["borderline_ghost"]
][[
    "image",
    "cell_id",
    "label",
    "dead_probability"
]]

combined.loc[
    combined["stage1_dead_prediction"] == "Dead",
    "final_prediction",
] = "Dead"

morph_pred_series = pd.Series(morph_cv_pred, index=morph_labeled.index)

combined.loc[
    morph_labeled.index,
    "stage2_morphology_prediction",
] = morph_pred_series

notdead_idx = combined["stage1_dead_prediction"] == "NotDead"

combined.loc[
    notdead_idx,
    "final_prediction",
] = combined.loc[
    notdead_idx,
    "stage2_morphology_prediction",
]

bad = combined["final_prediction"].isna()

if bad.sum() > 0:
    print("Warning: dropping rows with missing final prediction:", bad.sum())
    combined = combined.loc[~bad].copy()

combined["final_prediction"] = combined["final_prediction"].astype(str)
combined["label"] = combined["label"].astype(str)

labels = sorted(labeled["label"].astype(str).unique())

cm_final = confusion_matrix(
    combined["label"],
    combined["final_prediction"],
    labels=labels,
)

macro_f1 = f1_score(
    combined["label"],
    combined["final_prediction"],
    average="macro",
)

print("\nCombined Hierarchical Classifier")
print(classification_report(combined["label"], combined["final_prediction"]))
print("Macro F1:", macro_f1)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(cm_final, display_labels=labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(
    f"Hierarchical Confusion Matrix\nMacro F1 = {macro_f1:.3f}, Dead threshold = {DEAD_THRESHOLD}"
)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "hierarchical_confusion_matrix.png"), dpi=300)
plt.close()

combined.to_csv(
    os.path.join(DATA_DIR, "hierarchical_training_diagnostics.csv"),
    index=False,
)

def save_feature_importance(model, feature_names, out_path, title, top_n=20):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(out_path.replace(".png", ".csv"), index=False)

    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------------
# Train final models
# -------------------------

dead_model.fit(X_dead, y_dead)
morph_model.fit(X_morph, y_morph)

save_feature_importance(
    dead_model,
    dead_features,
    os.path.join(DATA_DIR, "stage1_dead_feature_importance.png"),
    "Stage 1 Dead Detector: Top 20 Features"
)

save_feature_importance(
    morph_model,
    feature_cols,
    os.path.join(DATA_DIR, "stage2_morphology_feature_importance.png"),
    "Stage 2 Morphology Classifier: Top 20 Features"
)

# -------------------------
# Predict unlabeled cells
# -------------------------

if len(unlabeled) > 0:
    analysis = unlabeled[["image", "cell_id"]].copy()

    X_unlabeled_dead = unlabeled[dead_features]
    dead_prob = dead_model.predict_proba(X_unlabeled_dead)

    dead_class_index = list(dead_model.classes_).index("Dead")
    analysis["prob_dead"] = dead_prob[:, dead_class_index]
    analysis["borderline_ghost"] = (
        (analysis["prob_dead"] >= BORDERLINE_LOW) &
        (analysis["prob_dead"] < DEAD_THRESHOLD)
    )

    analysis["stage1_prediction"] = np.where(
        analysis["prob_dead"] >= DEAD_THRESHOLD,
        "Dead",
        "NotDead",
    )

    analysis["final_prediction"] = "Dead"

    not_dead_mask = analysis["stage1_prediction"] == "NotDead"

    if not_dead_mask.sum() > 0:
        X_unlabeled_morph = unlabeled.loc[not_dead_mask, feature_cols]

        morph_pred = morph_model.predict(X_unlabeled_morph)
        morph_prob = morph_model.predict_proba(X_unlabeled_morph)

        analysis.loc[not_dead_mask, "final_prediction"] = morph_pred
        analysis.loc[not_dead_mask, "morph_confidence"] = morph_prob.max(axis=1)

        for i, cls in enumerate(morph_model.classes_):
            analysis.loc[not_dead_mask, f"prob_{cls}"] = morph_prob[:, i]

    analysis["review_flag"] = False
    analysis["review_reason"] = ""

    analysis.loc[
        (analysis["stage1_prediction"] == "Dead")
        & (analysis["prob_dead"] < DEAD_THRESHOLD + 0.10),
        "review_flag",
    ] = True

    analysis.loc[
        (analysis["stage1_prediction"] == "Dead")
        & (analysis["prob_dead"] < DEAD_THRESHOLD + 0.10),
        "review_reason",
    ] += "uncertain_dead; "

    if "morph_confidence" in analysis.columns:
        analysis.loc[
            (analysis["stage1_prediction"] == "NotDead")
            & (analysis["morph_confidence"] < 0.60),
            "review_flag",
        ] = True

        analysis.loc[
            (analysis["stage1_prediction"] == "NotDead")
            & (analysis["morph_confidence"] < 0.60),
            "review_reason",
        ] += "uncertain_morphology; "

    analysis.to_csv(
        os.path.join(DATA_DIR, "hierarchical_analysis_table.csv"),
        index=False,
    )

print("\nSaved:")
print(os.path.join(DATA_DIR, "stage1_dead_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "stage2_morphology_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "hierarchical_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "hierarchical_training_diagnostics.csv"))
print(os.path.join(DATA_DIR, "hierarchical_analysis_table.csv"))