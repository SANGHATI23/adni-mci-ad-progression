import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

# Load datasets
mri = pd.read_csv(DATA_DIR / "mri_features_clean.csv")
labels = pd.read_csv(DATA_DIR / "mci_conversion_labels.csv")

print("MRI shape:", mri.shape)
print("Labels shape:", labels.shape)

# Keep only baseline visit (IMPORTANT)
mri_bl = mri[mri["VISCODE"].isin(["bl", "sc"])]

print("Baseline MRI shape:", mri_bl.shape)

# Merge on RID
df = mri_bl.merge(labels, on="RID", how="inner")

print("Merged shape:", df.shape)

print("\nPreview:")
print(df.head())

# Save final dataset
output_path = DATA_DIR / "final_mri_dataset.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)