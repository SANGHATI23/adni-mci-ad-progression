import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

df = pd.read_csv(DATA_DIR / "progression_dataset.csv")

network_features = [
    "delta_VENTRICLES",
    "delta_LHIPPOC",
    "delta_RHIPPOC",
    "delta_LENTORHIN",
    "delta_RENTORHIN",
    "delta_LMIDTEMP",
    "delta_RMIDTEMP",
]

X = df[network_features]

corr = X.corr()

print("Brain-change correlation network")
print("--------------------------------")
print(corr)

output_path = DATA_DIR / "brain_change_correlation_network.csv"
corr.to_csv(output_path)

print("\nSaved:", output_path)