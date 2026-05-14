import pandas as pd
import numpy as np
from itertools import combinations

DATA_PATH = "data/processed/progression_dataset.csv"
OUT_PATH = "data/results/edge_statistical_validation.csv"

df = pd.read_csv(DATA_PATH)

features = [
    "delta_LHIPPOC", "delta_RHIPPOC",
    "delta_LENTORHIN", "delta_RENTORHIN",
    "delta_LMIDTEMP", "delta_RMIDTEMP",
    "delta_VENTRICLES"
]

label_col = "label"


def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    out = np.empty(n)
    out[order] = adjusted
    return out


def compute_correlations(data):
    return data.corr()


pmci = df[df[label_col] == "pMCI_36"][features].dropna()
smci = df[df[label_col] == "sMCI_36"][features].dropna()

print("pMCI:", len(pmci), "sMCI:", len(smci))

pmci_corr = compute_correlations(pmci)
smci_corr = compute_correlations(smci)

edges = list(combinations(features, 2))

combined = df[[label_col] + features].dropna().copy()
true_labels = combined[label_col].values

n_perm = 5000
results = []

np.random.seed(42)

for (f1, f2) in edges:

    obs_pmci = pmci_corr.loc[f1, f2]
    obs_smci = smci_corr.loc[f1, f2]
    obs_diff = obs_pmci - obs_smci

    perm_diffs = []

    for _ in range(n_perm):
        shuffled = np.random.permutation(true_labels)
        temp = combined.copy()
        temp["perm_label"] = shuffled

        p_data = temp[temp["perm_label"] == "pMCI_36"]
        s_data = temp[temp["perm_label"] == "sMCI_36"]

        if len(p_data) < 10 or len(s_data) < 10:
            continue

        p_corr = p_data[features].corr().loc[f1, f2]
        s_corr = s_data[features].corr().loc[f1, f2]

        if not np.isnan(p_corr) and not np.isnan(s_corr):
            perm_diffs.append(p_corr - s_corr)

    perm_diffs = np.array(perm_diffs)

    if len(perm_diffs) == 0:
        p_value = np.nan
    else:
        p_value = np.mean(np.abs(perm_diffs) >= abs(obs_diff))

    results.append({
        "edge": f"{f1}__{f2}",
        "pmci_corr": obs_pmci,
        "smci_corr": obs_smci,
        "difference_pmci_minus_smci": obs_diff,
        "permutation_p": p_value
    })


results_df = pd.DataFrame(results)

results_df["fdr_corrected_p"] = fdr_bh(
    results_df["permutation_p"].fillna(1.0).values
)

results_df["significant_fdr_0.05"] = results_df["fdr_corrected_p"] < 0.05

# sort by strongest difference
results_df = results_df.sort_values(
    by="difference_pmci_minus_smci",
    key=lambda x: abs(x),
    ascending=False
)

results_df.to_csv(OUT_PATH, index=False)

print("\n===== EDGE STATISTICAL VALIDATION =====")
print(results_df.head(15))

print(f"\nSaved: {OUT_PATH}")