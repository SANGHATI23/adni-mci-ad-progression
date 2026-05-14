import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report

DATA_DIR = Path("data/processed")

df = pd.read_csv(DATA_DIR / "final_mri_dataset.csv")

features = [
    "EICV",
    "VENTRICLES",
    "LHIPPOC",
    "RHIPPOC",
    "LENTORHIN",
    "RENTORHIN",
    "LMIDTEMP",
    "RMIDTEMP",
]

X = df[features]
y = df["label"].map({"sMCI_36": 0, "pMCI_36": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Baseline MRI Logistic Regression Results")
print("---------------------------------------")
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["sMCI_36", "pMCI_36"]))