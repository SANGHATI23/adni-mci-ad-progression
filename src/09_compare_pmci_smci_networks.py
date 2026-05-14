import pandas as pd
import numpy as np

# Load longitudinal progression dataset
df = pd.read_csv("data/processed/progression_dataset.csv")

delta_cols = [c for c in df.columns if c.startswith("delta_")]

print("Delta features used:")
print(delta_cols)

# Split groups
pmci = df[df["label"] == "pMCI_36"]
smci = df[df["label"] == "sMCI_36"]

print("\npMCI shape:", pmci.shape)
print("sMCI shape:", smci.shape)

# Correlation networks
pmci_corr = pmci[delta_cols].corr()
smci_corr = smci[delta_cols].corr()

# Difference network
diff_corr = pmci_corr - smci_corr

# Save outputs
pmci_corr.to_csv("data/results/pmci_brain_change_network.csv")
smci_corr.to_csv("data/results/smci_brain_change_network.csv")
diff_corr.to_csv("data/results/pmci_minus_smci_network_difference.csv")

print("\nSaved:")
print("data/results/pmci_brain_change_network.csv")
print("data/results/smci_brain_change_network.csv")
print("data/results/pmci_minus_smci_network_difference.csv")

print("\nTop network differences: ")

# Convert matrix to edge list
edges = []

for i in range(len(delta_cols)):
    for j in range(i + 1, len(delta_cols)):
        r1 = delta_cols[i]
        r2 = delta_cols[j]

        edges.append({
            "region_1": r1,
            "region_2": r2,
            "pmci_corr": pmci_corr.loc[r1, r2],
            "smci_corr": smci_corr.loc[r1, r2],
            "difference_pmci_minus_smci": diff_corr.loc[r1, r2],
            "abs_difference": abs(diff_corr.loc[r1, r2])
        })

edge_df = pd.DataFrame(edges)
edge_df = edge_df.sort_values("abs_difference", ascending=False)

edge_df.to_csv("data/results/top_network_differences.csv", index=False)

print(edge_df.head(15))