import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")

mri = pd.read_csv(DATA_DIR / "mri_features_clean.csv")
labels = pd.read_csv(DATA_DIR / "mci_conversion_labels.csv")

# Keep only baseline (sc) and 12-month (m12)
mri = mri[mri["VISCODE"].isin(["sc", "m12"])]

# Split
baseline = mri[mri["VISCODE"] == "sc"]
m12 = mri[mri["VISCODE"] == "m12"]

# Merge baseline with m12 on RID
df = baseline.merge(m12, on="RID", suffixes=("_bl", "_m12"))

print("Patients with both timepoints:", df.shape)

# Create delta features
features = [
    "VENTRICLES",
    "LHIPPOC", "RHIPPOC",
    "LENTORHIN", "RENTORHIN",
    "LMIDTEMP", "RMIDTEMP"
]

for f in features:
    df[f"delta_{f}"] = df[f"{f}_m12"] - df[f"{f}_bl"]

# Keep only delta features + RID
delta_cols = ["RID"] + [f"delta_{f}" for f in features]
df_delta = df[delta_cols]

print("\nDelta dataset shape:", df_delta.shape)
print("\nPreview:")
print(df_delta.head())

# Merge with labels
final = df_delta.merge(labels, on="RID", how="inner")

print("\nFinal progression dataset:", final.shape)

# Save
final.to_csv(OUTPUT_DIR / "progression_dataset.csv", index=False)

print("\nSaved: progression_dataset.csv")