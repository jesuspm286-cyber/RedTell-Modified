import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from lightgbm import LGBMClassifier


DEAD_THRESHOLD = 0.85


def get_feature_columns(df):
    drop_cols = ["image", "cell_id", "label", "predicted_label", "final_prediction"]
    return [c for c in df.columns if c not in drop_cols]


def get_dead_features(feature_cols):
    return [
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
            or "crenation_max_radial_deviation" in c.lower()
            or "crenation_radial_variation" in c.lower()
        )
    ]


def save_confusion_matrix(y_true, y_pred, labels, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{title}\nMacro F1 = {macro_f1:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return macro_f1


def main(train_dir, valid_dir):
    train_path = os.path.join(train_dir, "features.csv")
    valid_path = os.path.join(valid_dir, "features.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    train_labeled = train_df[train_df["label"].notna()].copy()
    valid_data = valid_df.copy()

    feature_cols = get_feature_columns(train_df)
    dead_features = get_dead_features(feature_cols)

    # Make sure validation has same columns
    missing = [c for c in feature_cols if c not in valid_data.columns]
    if missing:
        raise ValueError(f"Validation file is missing features: {missing}")

    # -------------------------
    # Stage 1: Dead detector
    # -------------------------

    X_dead_train = train_labeled[dead_features]
    y_dead_train = np.where(train_labeled["label"] == "Dead", "Dead", "NotDead")

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

    dead_model.fit(X_dead_train, y_dead_train)

    X_dead_valid = valid_data[dead_features]
    dead_prob = dead_model.predict_proba(X_dead_valid)

    dead_idx = list(dead_model.classes_).index("Dead")
    valid_data["prob_dead"] = dead_prob[:, dead_idx]

    valid_data["stage1_dead_prediction"] = np.where(
        valid_data["prob_dead"] >= DEAD_THRESHOLD,
        "Dead",
        "NotDead",
    )

    # -------------------------
    # Stage 2: Morphology classifier
    # -------------------------

    morph_train = train_labeled[train_labeled["label"] != "Dead"].copy()

    X_morph_train = morph_train[feature_cols]
    y_morph_train = morph_train["label"]

    # morph_model = RandomForestClassifier(
    #     n_estimators=300,
    #     random_state=42,
    #     class_weight="balanced",
    # )
    morph_model = LGBMClassifier(
        objective="multiclass",
        random_state=42,
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        n_jobs=-1,
    )

    morph_model.fit(X_morph_train, y_morph_train)

    valid_data["final_prediction"] = "Dead"

    notdead_mask = valid_data["stage1_dead_prediction"] == "NotDead"

    if notdead_mask.sum() > 0:
        X_morph_valid = valid_data.loc[notdead_mask, feature_cols]
        morph_pred = morph_model.predict(X_morph_valid)
        morph_prob = morph_model.predict_proba(X_morph_valid)

        valid_data.loc[notdead_mask, "final_prediction"] = morph_pred
        valid_data.loc[notdead_mask, "morph_confidence"] = morph_prob.max(axis=1)

        for i, cls in enumerate(morph_model.classes_):
            valid_data.loc[notdead_mask, f"prob_{cls}"] = morph_prob[:, i]

    # -------------------------
    # Save predictions
    # -------------------------

    out_csv = os.path.join(valid_dir, "hierarchical_validation_predictions.csv")
    valid_data.to_csv(out_csv, index=False)

    print("\nSaved predictions to:")
    print(out_csv)

    # -------------------------
    # If labels exist, evaluate
    # -------------------------

    if "label" in valid_data.columns and valid_data["label"].notna().any():
        eval_df = valid_data[valid_data["label"].notna()].copy()


        print("\nStage 1: Dead vs NotDead")
        y_true_dead = np.where(eval_df["label"] == "Dead", "Dead", "NotDead")
        y_pred_dead = eval_df["stage1_dead_prediction"]

        print(classification_report(y_true_dead, y_pred_dead))

        save_confusion_matrix(
            y_true_dead,
            y_pred_dead,
            ["Dead", "NotDead"],
            os.path.join(valid_dir, "validation_stage1_dead_confusion_matrix.png"),
            f"Validation Stage 1: Dead vs NotDead\nDead threshold = {DEAD_THRESHOLD}",
        )

        print("\nFinal hierarchical classifier")
        labels = sorted(eval_df["label"].astype(str).unique())

        print(classification_report(eval_df["label"], eval_df["final_prediction"]))

        macro_f1 = save_confusion_matrix(
            eval_df["label"],
            eval_df["final_prediction"],
            labels,
            os.path.join(valid_dir, "validation_hierarchical_confusion_matrix.png"),
            f"Validation Hierarchical Classifier\nDead threshold = {DEAD_THRESHOLD}",
        )

        print("Validation Macro F1:", macro_f1)

        # -------------------------
        # Save validation diagnostics
        # -------------------------

        eval_df["correct_prediction"] = (
            eval_df["label"].astype(str) == eval_df["final_prediction"].astype(str)
        )

        eval_df["stage1_correct"] = (
            np.where(eval_df["label"] == "Dead", "Dead", "NotDead")
            == eval_df["stage1_dead_prediction"]
        )

        diagnostic_cols = [
            "image",
            "cell_id",
            "label",
            "stage1_dead_prediction",
            "prob_dead",
            "final_prediction",
            "correct_prediction",
            "stage1_correct",
        ]

        extra_prob_cols = [c for c in eval_df.columns if c.startswith("prob_")]
        diagnostic_cols += extra_prob_cols

        diagnostic_cols = [c for c in diagnostic_cols if c in eval_df.columns]

        diagnostics = eval_df[diagnostic_cols].copy()

        diagnostics = diagnostics.sort_values(
            ["correct_prediction", "label", "final_prediction", "image", "cell_id"],
            ascending=[True, True, True, True, True],
        )

        diagnostics_path = os.path.join(valid_dir, "hierarchical_validation_diagnostics.csv")
        diagnostics.to_csv(diagnostics_path, index=False)

        print("\nSaved validation diagnostics to:")
        print(diagnostics_path)

        print("\nIncorrect predictions:")
        print(
            diagnostics[diagnostics["correct_prediction"] == False][
                [
                    "image",
                    "cell_id",
                    "label",
                    "final_prediction",
                    "prob_dead",
                ]
            ].to_string(index=False)
        )

    # -------------------------
    # Summary counts
    # -------------------------

    counts = valid_data["final_prediction"].value_counts()
    percents = valid_data["final_prediction"].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "count": counts,
        "percent": percents,
    })

    summary_path = os.path.join(valid_dir, "hierarchical_validation_summary.csv")
    summary.to_csv(summary_path)

    print("\nPrediction summary:")
    print(summary)
    print("\nSaved summary to:")
    print(summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default="Data_for_model",
        help="Folder containing labeled training features.csv",
    )

    parser.add_argument(
        "--valid",
        required=True,
        help="Folder containing validation features.csv",
    )

    args = parser.parse_args()

    main(args.train, args.valid)