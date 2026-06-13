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

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

data_dir = "Data"
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

model = GradientBoostingClassifier(
        random_state=42
)

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
cm = confusion_matrix(y, y_pred)
print(cm)
print(sorted(y.unique()))

# Confusion matrix figure
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title(f"Validation Confusion Matrix\nMacro F1 = {macro_f1:.3f}")
plt.tight_layout()
plt.savefig("validation_confusion_matrix.png", dpi=300)
plt.close()

# Feature importance figure
top_n = 20
top_features = importance.head(top_n).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("Feature importance")
ax.set_title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig("validation_feature_importance.png", dpi=300)
plt.close()