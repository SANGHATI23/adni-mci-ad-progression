import pyreadr
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/adni/ADNIMERGE2/data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load MRI data
file_path = DATA_DIR / "UCSDVOL.rda"
result = pyreadr.read_r(file_path)
df = result["UCSDVOL"]

print("Original shape:", df.shape)

# Keep only relevant columns
cols = [
    "RID", "VISCODE",
    "EICV", "VENTRICLES",
    "LHIPPOC", "RHIPPOC",
    "LENTORHIN", "RENTORHIN",
    "LMIDTEMP", "RMIDTEMP"
]

df = df[cols]

# Drop missing values
df = df.dropna()

print("After cleaning:", df.shape)

# Save
output_path = OUTPUT_DIR / "mri_features_clean.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nPreview:")
print(df.head())