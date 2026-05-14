import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/progression_dataset.csv"
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["event"] = np.where(df["label"] == "pMCI_36", 1, 0)

df["time_to_event"] = np.where(
    df["event"] == 1,
    df["conversion_month"],
    df["max_followup_month"]
)

feature_cols = [
    "delta_LHIPPOC",
    "delta_RHIPPOC",
    "delta_LENTORHIN",
    "delta_RENTORHIN",
    "delta_LMIDTEMP",
    "delta_RMIDTEMP",
    "delta_VENTRICLES"
]

cox_df = df[["time_to_event", "event"] + feature_cols].dropna().copy()

scaler = StandardScaler()
cox_df[feature_cols] = scaler.fit_transform(cox_df[feature_cols])

cph = CoxPHFitter(penalizer=0.1)
cph.fit(
    cox_df,
    duration_col="time_to_event",
    event_col="event"
)

summary = cph.summary.reset_index()
summary.to_csv(OUT_DIR / "cox_mri_delta_survival_results.csv", index=False)

print("\n===== COX SURVIVAL MODEL RESULTS =====")
print(summary[
    [
        "covariate",
        "coef",
        "exp(coef)",
        "p",
        "coef lower 95%",
        "coef upper 95%"
    ]
])

print("\nConcordance index:")
print(cph.concordance_index_)

print("\nSaved:")
print(OUT_DIR / "cox_mri_delta_survival_results.csv")