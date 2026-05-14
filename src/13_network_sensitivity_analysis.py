import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

DATA_PATH = "data/processed/progression_dataset.csv"
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Columns
# -----------------------------
region_cols = [
    "delta_LHIPPOC",
    "delta_RHIPPOC",
    "delta_LENTORHIN",
    "delta_RENTORHIN",
    "delta_LMIDTEMP",
    "delta_RMIDTEMP",
    "delta_VENTRICLES"
]

label_col = "label"

pmci = df[df[label_col] == "pMCI_36"].copy()
smci = df[df[label_col] == "sMCI_36"].copy()

print("pMCI:", pmci.shape[0])
print("sMCI:", smci.shape[0])

# -----------------------------
# Helper: correlation edge table
# -----------------------------
def compute_edges(data, cols):
    results = []

    for a, b in combinations(cols, 2):
        r, _ = pearsonr(data[a], data[b])
        results.append({
            "edge": f"{a}__{b}",
            "corr": r,
            "abs_corr": abs(r)
        })

    return pd.DataFrame(results)


pmci_edges = compute_edges(pmci, region_cols).rename(
    columns={"corr": "pmci_corr", "abs_corr": "pmci_abs_corr"}
)

smci_edges = compute_edges(smci, region_cols).rename(
    columns={"corr": "smci_corr", "abs_corr": "smci_abs_corr"}
)

edge_df = pmci_edges.merge(smci_edges, on="edge")
edge_df["diff"] = edge_df["pmci_corr"] - edge_df["smci_corr"]
edge_df["abs_diff"] = edge_df["diff"].abs()

# -----------------------------
# Permutation test for edge difference
# -----------------------------
def permutation_edge_pvalue(df, edge_name, cols, label_col, n_perm=1000, seed=42):
    np.random.seed(seed)

    a, b = edge_name.split("__")

    observed_pmci = df[df[label_col] == "pMCI_36"]
    observed_smci = df[df[label_col] == "sMCI_36"]

    obs_r_pmci = pearsonr(observed_pmci[a], observed_pmci[b])[0]
    obs_r_smci = pearsonr(observed_smci[a], observed_smci[b])[0]
    obs_diff = obs_r_pmci - obs_r_smci

    perm_diffs = []

    labels = df[label_col].values.copy()

    for _ in range(n_perm):
        np.random.shuffle(labels)
        temp = df.copy()
        temp["perm_label"] = labels

        perm_pmci = temp[temp["perm_label"] == "pMCI_36"]
        perm_smci = temp[temp["perm_label"] == "sMCI_36"]

        r_pmci = pearsonr(perm_pmci[a], perm_pmci[b])[0]
        r_smci = pearsonr(perm_smci[a], perm_smci[b])[0]

        perm_diffs.append(r_pmci - r_smci)

    perm_diffs = np.array(perm_diffs)

    p_value = np.mean(np.abs(perm_diffs) >= abs(obs_diff))

    return obs_diff, p_value


# -----------------------------
# Sensitivity settings
# -----------------------------
thresholds = [0.20, 0.30, 0.40]
permutation_counts = [1000]

all_results = []

for threshold in thresholds:
    top_n = int(np.ceil(len(edge_df) * threshold))

    selected_edges = edge_df.sort_values("abs_diff", ascending=False).head(top_n)

    for n_perm in permutation_counts:
        print(f"\nRunning threshold={threshold}, permutations={n_perm}")

        temp_results = []

        for edge in selected_edges["edge"]:
            obs_diff, p = permutation_edge_pvalue(
                df=df,
                edge_name=edge,
                cols=region_cols,
                label_col=label_col,
                n_perm=n_perm,
                seed=42
            )

            base_row = edge_df[edge_df["edge"] == edge].iloc[0]

            temp_results.append({
                "threshold_top_percent": threshold,
                "n_permutations": n_perm,
                "edge": edge,
                "pmci_corr": base_row["pmci_corr"],
                "smci_corr": base_row["smci_corr"],
                "difference_pmci_minus_smci": obs_diff,
                "raw_p_value": p
            })

        temp_df = pd.DataFrame(temp_results)

        temp_df["fdr_corrected_p"] = multipletests(
            temp_df["raw_p_value"],
            method="fdr_bh"
        )[1]

        temp_df["significant_fdr_0.05"] = temp_df["fdr_corrected_p"] < 0.05

        all_results.append(temp_df)

final_results = pd.concat(all_results, ignore_index=True)

# -----------------------------
# Directional stability summary
# -----------------------------
final_results["direction"] = np.where(
    final_results["difference_pmci_minus_smci"] > 0,
    "higher_in_pMCI",
    "lower_in_pMCI"
)

direction_summary = (
    final_results
    .groupby("edge")
    .agg(
        times_tested=("edge", "count"),
        mean_difference=("difference_pmci_minus_smci", "mean"),
        direction_consistency=("direction", lambda x: x.value_counts().iloc[0] / len(x)),
        dominant_direction=("direction", lambda x: x.value_counts().index[0]),
        min_raw_p=("raw_p_value", "min"),
        min_fdr_p=("fdr_corrected_p", "min"),
        any_fdr_significant=("significant_fdr_0.05", "any")
    )
    .reset_index()
    .sort_values(["direction_consistency", "min_raw_p"], ascending=[False, True])
)

# -----------------------------
# Save outputs
# -----------------------------
final_results.to_csv(OUT_DIR / "network_sensitivity_edge_results.csv", index=False)
direction_summary.to_csv(OUT_DIR / "network_sensitivity_direction_summary.csv", index=False)

print("\nSaved:")
print(OUT_DIR / "network_sensitivity_edge_results.csv")
print(OUT_DIR / "network_sensitivity_direction_summary.csv")

print("\nTop directionally stable edges:")
print(direction_summary.head(15))