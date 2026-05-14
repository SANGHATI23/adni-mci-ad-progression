import pandas as pd
import numpy as np
import networkx as nx

# Load networks
pmci = pd.read_csv("data/results/pmci_brain_change_network.csv", index_col=0)
smci = pd.read_csv("data/results/smci_brain_change_network.csv", index_col=0)

THRESHOLD = 0.4  # keep meaningful edges

def build_graph(corr_matrix):
    G = nx.Graph()

    nodes = corr_matrix.columns

    for node in nodes:
        G.add_node(node)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            r1 = nodes[i]
            r2 = nodes[j]

            weight = corr_matrix.loc[r1, r2]

            if abs(weight) >= THRESHOLD:
                G.add_edge(
                    r1, r2,
                    weight=abs(weight)  # use magnitude
                )

    return G

def compute_metrics(G, name):
    print(f"\n===== {name} NETWORK =====")

    # Node metrics
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    centrality = nx.degree_centrality(G)

    node_df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "strength": [strength[n] for n in G.nodes()],
        "centrality": [centrality[n] for n in G.nodes()]
    }).sort_values("strength", ascending=False)

    print("\nTop nodes by strength:")
    print(node_df.head())

    # Global metrics
    density = nx.density(G)

    clustering = nx.average_clustering(G, weight="weight")

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G, weight="weight")
    else:
        path_length = np.nan

    efficiency = nx.global_efficiency(G)

    print("\nGlobal metrics:")
    print(f"Density: {density:.3f}")
    print(f"Clustering: {clustering:.3f}")
    print(f"Path length: {path_length}")
    print(f"Global efficiency: {efficiency:.3f}")

    return node_df, {
        "density": density,
        "clustering": clustering,
        "path_length": path_length,
        "efficiency": efficiency
    }

# Build graphs
G_pmci = build_graph(pmci)
G_smci = build_graph(smci)

# Compute metrics
pmci_nodes, pmci_global = compute_metrics(G_pmci, "pMCI")
smci_nodes, smci_global = compute_metrics(G_smci, "sMCI")

# Save
pmci_nodes.to_csv("data/results/pmci_node_metrics.csv", index=False)
smci_nodes.to_csv("data/results/smci_node_metrics.csv", index=False)

global_df = pd.DataFrame([
    {"group": "pMCI", **pmci_global},
    {"group": "sMCI", **smci_global}
])

global_df.to_csv("data/results/network_global_metrics.csv", index=False)

print("\nSaved:")
print("data/results/pmci_node_metrics.csv")
print("data/results/smci_node_metrics.csv")
print("data/results/network_global_metrics.csv")