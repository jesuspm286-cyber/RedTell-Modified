import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True, help="Training features.csv")
parser.add_argument("--test", required=True, help="Validation features.csv")
parser.add_argument("--out", default="validation_results", help="Output folder")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

train_df = pd.read_csv(args.train)
test_df = pd.read_csv(args.test)

train_labeled = train_df[train_df["label"].notna()].copy()
test_labeled = test_df[test_df["label"].notna()].copy()

drop_cols = ["image", "cell_id", "label", "predicted_label"]

feature_cols = [
    c for c in train_df.columns
    if c not in drop_cols and c in test_df.columns
]

X_train = train_labeled[feature_cols]
y_train = train_labeled["label"]

X_test = test_labeled[feature_cols]
y_test = test_labeled["label"]

#model = GradientBoostingClassifier(random_state=42)
model = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_estimators=600,
    learning_rate=0.06,
    num_leaves=50,
    max_depth=4,
    n_jobs=-1
)

# model = RandomForestClassifier(
#     n_estimators=300,
#     random_state=42,
#     class_weight="balanced"
# )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

labels = sorted(y_train.unique())
macro_f1 = f1_score(y_test, y_pred, average="macro")

print()
print("Labels:", labels)
print()
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(cm)
print()
print(classification_report(y_test, y_pred))

# Save predictions
test_labeled["predicted_label"] = y_pred
test_labeled.to_csv(os.path.join(args.out, "validation_predictions.csv"), index=False)

# Feature importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(os.path.join(args.out, "validation_feature_importance.csv"), index=False)

# Confusion matrix figure
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Validation Confusion Matrix\nMacro F1 = {macro_f1:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "validation_confusion_matrix.png"), dpi=300)
plt.close()

# Feature importance figure
top_n = 20
top_features = importance.head(top_n).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Feature importance")
ax.set_title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "validation_feature_importance.png"), dpi=300)
plt.close()

print("Saved results to:", args.out)