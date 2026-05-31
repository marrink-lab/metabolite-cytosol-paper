#!/usr/bin/env python3
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("../../mystyle.mplstyle")

CLASS_COLORS = {
    "Nucleotides": "#7443BB",
    "Amino_Acids": "#F9B714",
    "Carbohydrates": "#3FB760",
    "Cofactors": "#B7C61E",
    "Fatty_Acids": "#05938E",
    "Ions": "#5B81F6",
    "Other_Metabolites": "#CC7FDB",
}
CLASS_LABELS = {
    "Amino_Acids": "Amino acids",
    "Carbohydrates": "Carbohydrates",
    "Cofactors": "Cofactors",
    "Ions": "Ions",
    "Fatty_Acids": "Fatty acids",
    "Nucleotides": "Nucleotides",
    "Other_Metabolites": "Other",
}

mol_to_class = dict(
    zip(*pd.read_csv("molecules_list.csv")[["resname", "class"]].values.T)
)


def get_class(rn):
    if rn not in mol_to_class:
        print(f"Warning: resname '{rn}' not found in molecules_list.csv, skipping.")
        return None
    return mol_to_class[rn]


# ── Per-cluster class fractions, zeros included for absent classes ────────────
cluster_class_fracs = defaultdict(list)

for rep_idx in range(3):
    with open(f"cluster_compositions_rep{rep_idx}.pkl", "rb") as f:
        data = pickle.load(f)
    for comp in data["cluster_compositions"]:
        class_counts = defaultdict(int)
        for rn, count in comp.items():
            cls = get_class(rn)
            if cls:
                class_counts[cls] += count
        total = sum(class_counts.values())
        if total == 0:
            continue
        # Record a fraction for every known class. Each cluster contributes a
        # probability vector summing to 1, so per-class means also sum to 1.
        for cls in CLASS_COLORS:
            cluster_class_fracs[cls].append(class_counts.get(cls, 0) / total)

mean_fracs = {cls: np.mean(vals) for cls, vals in cluster_class_fracs.items()}
mean_fracs = {
    cls: v for cls, v in sorted(mean_fracs.items(), key=lambda x: -x[1]) if v > 0.001
}

# ── Plot: donut chart ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))

wedges, _ = ax.pie(
    list(mean_fracs.values()),
    colors=[CLASS_COLORS[cls] for cls in mean_fracs],
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.2, width=0.3),
)

ax.legend(
    wedges,
    [f"{CLASS_LABELS.get(cls, cls)}:  {v * 100:.0f}%" for cls, v in mean_fracs.items()],
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    fontsize=12,
    frameon=False,
    handlelength=1.2,
    handletextpad=0.5,
)

fig.tight_layout(pad=0.3)
fig.savefig("fig3A.png", dpi=300, bbox_inches="tight", transparent=True)
