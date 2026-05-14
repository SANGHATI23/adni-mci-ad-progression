import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/results/pmci_minus_smci_network_difference.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FIGURE_DIR / "figure_05_pmci_minus_smci_difference_heatmap.png"

# -----------------------------
# Load difference matrix
# -----------------------------
diff = pd.read_csv(INPUT_PATH, index_col=0)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    diff,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Correlation difference (pMCI - sMCI)"}
)

plt.title(
    "Difference Heatmap of Brain-Change Networks\npMCI minus sMCI",
    fontsize=16,
    pad=18
)

plt.xlabel("Brain Region Change Feature")
plt.ylabel("Brain Region Change Feature")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved difference heatmap to: {OUTPUT_PATH}")