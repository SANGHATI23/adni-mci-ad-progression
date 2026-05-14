import pandas as pd
import numpy as np
import networkx as nx

DATA_PATH = "data/processed/progression_dataset.csv"
OUT_PATH = "data/results/statistical_validation_network.csv"

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


def graph_metrics(data, threshold=0.30):
    corr = data.corr().abs()
    G = nx.Graph()

    for node in corr.columns:
        G.add_node(node)

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            w = corr.iloc[i, j]

            if pd.notna(w) and w >= threshold:
                G.add_edge(
                    corr.columns[i],
                    corr.columns[j],
                    weight=w,
                    distance=1 / w
                )

    density = nx.density(G)

    if G.number_of_edges() > 0:
        clustering = nx.average_clustering(G, weight="weight")
    else:
        clustering = 0.0

    if G.number_of_edges() > 0:
        if nx.is_connected(G):
            path_length = nx.average_shortest_path_length(G, weight="distance")
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            path_length = nx.average_shortest_path_length(subG, weight="distance")
    else:
        path_length = np.nan

    efficiency = nx.global_efficiency(G)

    return {
        "density": density,
        "clustering": clustering,
        "path_length": path_length,
        "global_efficiency": efficiency
    }


pmci = df[df[label_col] == "pMCI_36"][features].dropna()
smci = df[df[label_col] == "sMCI_36"][features].dropna()

print("pMCI sample size:", len(pmci))
print("sMCI sample size:", len(smci))

observed_pmci = graph_metrics(pmci)
observed_smci = graph_metrics(smci)

observed_diff = {
    metric: observed_pmci[metric] - observed_smci[metric]
    for metric in observed_pmci.keys()
}

combined = df[[label_col] + features].dropna().copy()
true_labels = combined[label_col].values

n_perm = 5000
results = []

np.random.seed(42)

for metric in observed_diff.keys():
    perm_diffs = []

    for _ in range(n_perm):
        shuffled_labels = np.random.permutation(true_labels)

        temp = combined.copy()
        temp["perm_label"] = shuffled_labels

        p_data = temp[temp["perm_label"] == "pMCI_36"][features]
        s_data = temp[temp["perm_label"] == "sMCI_36"][features]

        if len(p_data) < 10 or len(s_data) < 10:
            continue

        p_metric = graph_metrics(p_data)[metric]
        s_metric = graph_metrics(s_data)[metric]

        if not np.isnan(p_metric) and not np.isnan(s_metric):
            perm_diffs.append(p_metric - s_metric)

    perm_diffs = np.array(perm_diffs)

    if len(perm_diffs) == 0:
        p_value = np.nan
    else:
        p_value = np.mean(np.abs(perm_diffs) >= abs(observed_diff[metric]))

    results.append({
        "metric": metric,
        "pmci": observed_pmci[metric],
        "smci": observed_smci[metric],
        "observed_difference_pmci_minus_smci": observed_diff[metric],
        "permutation_p_value": p_value
    })

results_df = pd.DataFrame(results)

results_df["fdr_corrected_p"] = fdr_bh(
    results_df["permutation_p_value"].fillna(1.0).values
)

results_df["significant_fdr_0.05"] = results_df["fdr_corrected_p"] < 0.05

results_df.to_csv(OUT_PATH, index=False)

print("\n===== STATISTICAL VALIDATION RESULTS =====")
print(results_df)

print(f"\nSaved: {OUT_PATH}")