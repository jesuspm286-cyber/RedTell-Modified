import argparse
import os
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

DATA_DIR = args.data

######### Change the threshold for dead detection as needed ###############

DEAD_THRESHOLD = 0.994

# Feature-selection settings. Selection is performed separately for each stage.
USE_FEATURE_SELECTION = True
CORRELATION_THRESHOLD = 0.95
N_DEAD_FEATURES = 20
N_MORPH_FEATURES = 30


BIOLOGICAL_LABELS = ["Viable", "Crenated", "Dead"]


def to_biological_label(label):
    """Combine Healthy and Stressed into the final Viable class."""
    if label in {"Healthy", "Stressed"}:
        return "Viable"
    return label


def select_reduced_features(
    X,
    y,
    n_features,
    corr_threshold=0.95,
):
    """Remove unusable/correlated columns, then select features with RFE.

    The returned feature names are used by the existing hierarchical models;
    this function does not replace either classifier.
    """
    X_numeric = X.apply(pd.to_numeric, errors="coerce")

    usable_columns = [
        column
        for column in X_numeric.columns
        if not X_numeric[column].isna().all()
        and X_numeric[column].nunique(dropna=True) > 1
    ]

    if not usable_columns:
        raise ValueError("No usable numerical features were found.")

    X_numeric = X_numeric[usable_columns]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_numeric),
        columns=usable_columns,
        index=X_numeric.index,
    )

    correlation = X_imputed.corr().abs()
    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )

    correlated_features = [
        column
        for column in upper_triangle.columns
        if (upper_triangle[column] > corr_threshold).any()
    ]

    retained_features = [
        column
        for column in usable_columns
        if column not in correlated_features
    ]

    if not retained_features:
        raise ValueError(
            "Correlation filtering removed every candidate feature. "
            "Increase CORRELATION_THRESHOLD."
        )

    X_filtered = X_imputed[retained_features]
    number_to_select = min(n_features, X_filtered.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    selector = RFE(
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=42,
        ),
        n_features_to_select=number_to_select,
        step=max(1, X_filtered.shape[1] // 20),
    )
    selector.fit(X_scaled, y)

    selected_features = [
        feature
        for feature, selected in zip(retained_features, selector.support_)
        if selected
    ]

    return selected_features, correlated_features


def save_selected_features(
    selected_features,
    correlated_features,
    stage_name,
):
    """Save the fixed feature list used by one hierarchy stage."""
    output = {
        "stage": stage_name,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "correlated_features_removed": correlated_features,
    }

    output_path = os.path.join(
        DATA_DIR,
        f"{stage_name}_selected_features.json",
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Saved selected features to: {output_path}")

features_path = os.path.join(DATA_DIR, "features.csv")
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
        # or "red" in c.lower()
        # or "green" in c.lower()
        or "blue" in c.lower()
        or "rgb" in c.lower()
        or "saturation" in c.lower()
    )
]

# dead_features = [
#     "cell_red_fraction_vs_bg",
#     "cell_rgb_saturation_proxy",
#     "cell_red_mean",
#     "cell_red_blue_ratio",
#     "cell_red_to_bg_red_ratio",
# ]

y_dead = np.where(labeled["label"] == "Dead", "Dead", "NotDead")

if USE_FEATURE_SELECTION:
    dead_features, dead_correlated_removed = select_reduced_features(
        X=labeled[dead_features],
        y=y_dead,
        n_features=N_DEAD_FEATURES,
        corr_threshold=CORRELATION_THRESHOLD,
    )
    save_selected_features(
        selected_features=dead_features,
        correlated_features=dead_correlated_removed,
        stage_name="stage1_dead",
    )

print(f"Stage 1 dead features ({len(dead_features)} selected):")
for feature in dead_features:
    print(" ", feature)

X_dead = labeled[dead_features]

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

# dead_model = LogisticRegression(
#     class_weight="balanced",
#     max_iter=5000,
#     random_state=42,
# )

# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# dead_model = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("classifier", LogisticRegression(
#         class_weight="balanced",
#         max_iter=5000,
#         random_state=42,
#     ))
# ])

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
y_morph = morph_labeled["label"]

morph_features = feature_cols.copy()

if USE_FEATURE_SELECTION:
    morph_features, morph_correlated_removed = select_reduced_features(
        X=morph_labeled[morph_features],
        y=y_morph,
        n_features=N_MORPH_FEATURES,
        corr_threshold=CORRELATION_THRESHOLD,
    )
    save_selected_features(
        selected_features=morph_features,
        correlated_features=morph_correlated_removed,
        stage_name="stage2_morphology",
    )

print(f"\nStage 2 morphology features ({len(morph_features)} selected):")
for feature in morph_features:
    print(" ", feature)

X_morph = morph_labeled[morph_features]

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

morph_cv_prob = cross_val_predict(
    morph_model,
    X_morph,
    y_morph,
    cv=5,
    method="predict_proba"
)

morph_classes = list(morph_model.fit(X_morph, y_morph).classes_)

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

# Add Stage 2 morphology probabilities
for j, cls in enumerate(morph_classes):
    combined.loc[morph_labeled.index, f"prob_morph_{cls}"] = morph_cv_prob[:, j]

combined["morph_confidence"] = np.nan
combined.loc[morph_labeled.index, "morph_confidence"] = morph_cv_prob.max(axis=1)

combined.loc[
    morph_labeled.index,
    "stage2_morphology_prediction",
] = morph_pred_series

# True Dead cells missed by Stage 1 were excluded from Stage 2 CV because
# Stage 2 is trained only on non-Dead labels. Predict those missed Dead cells
# with the fitted morphology model so they remain in the diagnostics table.
missed_dead_indices = combined.index[
    (combined["label"] == "Dead")
    & (combined["stage1_dead_prediction"] == "NotDead")
]

if len(missed_dead_indices) > 0:
    X_missed_dead_morph = labeled.loc[
        missed_dead_indices,
        morph_features,
    ]

    missed_dead_morph_pred = morph_model.predict(
        X_missed_dead_morph
    )
    missed_dead_morph_prob = morph_model.predict_proba(
        X_missed_dead_morph
    )

    combined.loc[
        missed_dead_indices,
        "stage2_morphology_prediction",
    ] = missed_dead_morph_pred

    combined.loc[
        missed_dead_indices,
        "morph_confidence",
    ] = missed_dead_morph_prob.max(axis=1)

    for j, cls in enumerate(morph_model.classes_):
        combined.loc[
            missed_dead_indices,
            f"prob_morph_{cls}",
        ] = missed_dead_morph_prob[:, j]

notdead_idx = combined["stage1_dead_prediction"] == "NotDead"

combined.loc[
    notdead_idx,
    "final_prediction",
] = combined.loc[
    notdead_idx,
    "stage2_morphology_prediction",
]

# # --------------------------------------------------
# # Let Stage 2 overrule Stage 1 for uncertain dead cells
# # --------------------------------------------------

# OVERRULE_DEAD = 0.9995
# MORPH_CONFIDENCE = 0.60

# dead_mask = combined["stage1_dead_prediction"] == "Dead"

# combined.loc[
#     dead_mask &
#     (combined["dead_probability"] < OVERRULE_DEAD) &
#     (combined["morph_confidence"] >= MORPH_CONFIDENCE),
#     "final_prediction"
# ] = combined.loc[
#     dead_mask &
#     (combined["dead_probability"] < OVERRULE_DEAD) &
#     (combined["morph_confidence"] >= MORPH_CONFIDENCE),
#     "stage2_morphology_prediction"
# ]

bad = combined["final_prediction"].isna()

if bad.sum() > 0:
    print(
        "Warning: cells still missing final predictions:",
        int(bad.sum()),
    )
    print(
        combined.loc[
            bad,
            [
                "image",
                "cell_id",
                "label",
                "stage1_dead_prediction",
                "stage2_morphology_prediction",
            ],
        ].to_string(index=False)
    )
    raise RuntimeError(
        "Some labeled cells are missing final predictions. "
        "They were not dropped; inspect the rows printed above."
    )

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

# -------------------------
# Biological evaluation
# Healthy + Stressed = Viable
# -------------------------

combined["biological_label"] = combined["label"].map(to_biological_label)
combined["biological_prediction"] = combined["final_prediction"].map(
    to_biological_label
)

bio_cm = confusion_matrix(
    combined["biological_label"],
    combined["biological_prediction"],
    labels=BIOLOGICAL_LABELS,
)

bio_macro_f1 = f1_score(
    combined["biological_label"],
    combined["biological_prediction"],
    labels=BIOLOGICAL_LABELS,
    average="macro",
)

bio_accuracy = (
    combined["biological_label"]
    == combined["biological_prediction"]
).mean()

print("\nBiological Hierarchical Classifier")
print(
    classification_report(
        combined["biological_label"],
        combined["biological_prediction"],
        labels=BIOLOGICAL_LABELS,
        zero_division=0,
    )
)
print("Biological accuracy:", bio_accuracy)
print("Biological macro F1:", bio_macro_f1)

bio_report = pd.DataFrame(
    classification_report(
        combined["biological_label"],
        combined["biological_prediction"],
        labels=BIOLOGICAL_LABELS,
        output_dict=True,
        zero_division=0,
    )
).T

bio_report.to_csv(
    os.path.join(DATA_DIR, "hierarchical_biological_classification_report.csv")
)

bio_cm_df = pd.DataFrame(
    bio_cm,
    index=BIOLOGICAL_LABELS,
    columns=BIOLOGICAL_LABELS,
)
bio_cm_df.to_csv(
    os.path.join(DATA_DIR, "hierarchical_biological_confusion_matrix_counts.csv")
)

bio_cm_norm = bio_cm_df.div(
    bio_cm_df.sum(axis=1).replace(0, np.nan),
    axis=0,
)
bio_cm_norm.to_csv(
    os.path.join(DATA_DIR, "hierarchical_biological_confusion_matrix_normalized.csv")
)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    bio_cm,
    display_labels=BIOLOGICAL_LABELS,
)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(
    "Biological Hierarchical Confusion Matrix\n"
    f"Accuracy = {bio_accuracy:.3f}, Macro F1 = {bio_macro_f1:.3f}"
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        DATA_DIR,
        "hierarchical_biological_confusion_matrix.png",
    ),
    dpi=300,
)
plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    bio_cm_norm.to_numpy(),
    display_labels=BIOLOGICAL_LABELS,
)
disp.plot(ax=ax, values_format=".2f", colorbar=False)
ax.set_title(
    "Normalized Biological Hierarchical Confusion Matrix\n"
    "Healthy + Stressed = Viable"
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        DATA_DIR,
        "hierarchical_biological_confusion_matrix_normalized.png",
    ),
    dpi=300,
)
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
    morph_features,
    os.path.join(DATA_DIR, "stage2_morphology_feature_importance.png"),
    "Stage 2 Morphology Classifier: Top 20 Features"
)

# plt.scatter(
#     X_dead["cell_red_fraction_vs_bg"],
#     X_dead["cell_rgb_saturation_proxy"],
#     c=(y_dead=="Dead"),
#     cmap="coolwarm"
# )

# -------------------------
# Predict unlabeled cells
# -------------------------

if len(unlabeled) > 0:

    prediction_df = df.copy()
    # analysis = prediction_df[["image", "cell_id", "label"]].copy()
    analysis = unlabeled[["image", "cell_id"]].copy()

    X_unlabeled_dead = unlabeled[dead_features]
    # X_unlabeled_dead = prediction_df[dead_features]

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
        X_unlabeled_morph = unlabeled.loc[not_dead_mask, morph_features]

        morph_pred = morph_model.predict(X_unlabeled_morph)
        morph_prob = morph_model.predict_proba(X_unlabeled_morph)

        analysis.loc[not_dead_mask, "final_prediction"] = morph_pred
        analysis.loc[not_dead_mask, "morph_confidence"] = morph_prob.max(axis=1)

        for i, cls in enumerate(morph_model.classes_):
            analysis.loc[not_dead_mask, f"prob_{cls}"] = morph_prob[:, i]

    # # --------------------------------------------------
    # # Allow morphology to overrule uncertain dead calls
    # # --------------------------------------------------

    # analysis.loc[
    #     (analysis["stage1_prediction"] == "Dead") &
    #     (analysis["prob_dead"] < OVERRULE_DEAD) &
    #     (analysis["morph_confidence"] >= MORPH_CONFIDENCE),
    #     "final_prediction"
    # ] = analysis.loc[
    #     (analysis["stage1_prediction"] == "Dead") &
    #     (analysis["prob_dead"] < OVERRULE_DEAD) &
    #     (analysis["morph_confidence"] >= MORPH_CONFIDENCE),
    #     "stage2_prediction"
    # ]

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

# combined.to_csv(os.path.join(DATA_DIR, "hierarchical_training_diagnostics.csv"), index=False)

print("\nSaved:")
print(os.path.join(DATA_DIR, "stage1_dead_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "stage2_morphology_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "hierarchical_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "hierarchical_biological_confusion_matrix.png"))
print(os.path.join(DATA_DIR, "hierarchical_biological_confusion_matrix_normalized.png"))
print(os.path.join(DATA_DIR, "hierarchical_biological_classification_report.csv"))
print(os.path.join(DATA_DIR, "hierarchical_training_diagnostics.csv"))
print(os.path.join(DATA_DIR, "hierarchical_analysis_table.csv"))