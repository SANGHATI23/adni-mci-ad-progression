import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

RESULTS_DIR = Path("data/results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

pmci = pd.read_csv(RESULTS_DIR / "pmci_brain_change_network.csv", index_col=0)
smci = pd.read_csv(RESULTS_DIR / "smci_brain_change_network.csv", index_col=0)
diff = pd.read_csv(RESULTS_DIR / "pmci_minus_smci_network_difference.csv", index_col=0)

# Cleaner labels
label_map = {
    "delta_VENTRICLES": "Ventricles",
    "delta_LHIPPOC": "L Hippocampus",
    "delta_RHIPPOC": "R Hippocampus",
    "delta_LENTORHIN": "L Entorhinal",
    "delta_RENTORHIN": "R Entorhinal",
    "delta_LMIDTEMP": "L Mid Temporal",
    "delta_RMIDTEMP": "R Mid Temporal",
}

def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(9, 7))
    mat = matrix.rename(index=label_map, columns=label_map)

    plt.imshow(mat, vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()

def plot_difference_heatmap(matrix, title, filename):
    plt.figure(figsize=(9, 7))
    mat = matrix.rename(index=label_map, columns=label_map)

    vmax = np.nanmax(np.abs(mat.values))
    plt.imshow(mat, vmin=-vmax, vmax=vmax)
    plt.colorbar(label="pMCI correlation - sMCI correlation")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()

def build_graph(corr_matrix, threshold=0.45):
    G = nx.Graph()

    for node in corr_matrix.columns:
        G.add_node(label_map.get(node, node))

    for i, r1 in enumerate(corr_matrix.columns):
        for j, r2 in enumerate(corr_matrix.columns):
            if j <= i:
                continue

            weight = corr_matrix.loc[r1, r2]

            if abs(weight) >= threshold:
                G.add_edge(
                    label_map.get(r1, r1),
                    label_map.get(r2, r2),
                    weight=weight,
                    abs_weight=abs(weight)
                )

    return G

def plot_network(corr_matrix, title, filename, threshold=0.45):
    G = build_graph(corr_matrix, threshold=threshold)

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    edge_widths = [G[u][v]["abs_weight"] * 5 for u, v in G.edges()]
    edge_labels = {
        (u, v): f'{G[u][v]["weight"]:.2f}'
        for u, v in G.edges()
    }

    positive_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] > 0]
    negative_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] < 0]

    nx.draw_networkx_nodes(G, pos, node_size=2500)
    nx.draw_networkx_labels(G, pos, font_size=9)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=positive_edges,
        width=[G[u][v]["abs_weight"] * 5 for u, v in positive_edges],
        style="solid"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=negative_edges,
        width=[G[u][v]["abs_weight"] * 5 for u, v in negative_edges],
        style="dashed"
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()

# Heatmaps
plot_heatmap(pmci, "pMCI Brain-Change Network", "pmci_network_heatmap.png")
plot_heatmap(smci, "sMCI Brain-Change Network", "smci_network_heatmap.png")
plot_difference_heatmap(
    diff,
    "Network Difference: pMCI minus sMCI",
    "pmci_minus_smci_difference_heatmap.png"
)

# Network graphs
plot_network(pmci, "pMCI Brain-Change Network", "pmci_network_graph.png")
plot_network(smci, "sMCI Brain-Change Network", "smci_network_graph.png")

print("Saved figures:")
print("figures/pmci_network_heatmap.png")
print("figures/smci_network_heatmap.png")
print("figures/pmci_minus_smci_difference_heatmap.png")
print("figures/pmci_network_graph.png")
print("figures/smci_network_graph.png")