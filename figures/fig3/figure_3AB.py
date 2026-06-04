#!/usr/bin/env python3
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("../../mystyle.mplstyle")

PICKLES = [
    "../../processed_data/cluster_composition/cluster_compositions_rep0.pkl",
    "../../processed_data/cluster_composition/cluster_compositions_rep1.pkl",
    "../../processed_data/cluster_composition/cluster_compositions_rep2.pkl",
]

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

TEAL = "#05938E"

mol_to_class = dict(
    zip(*pd.read_csv("../../processed_data/info.csv")[["resname", "class"]].values.T)
)


def get_class(rn):
    if rn not in mol_to_class:
        print(f"Warning: resname '{rn}' not found in molecules_list.csv, skipping.")
        return None
    return mol_to_class[rn]


# ── Load data in a single pass ───────────────────────────────────────────────
sizes = []
cluster_class_fracs = defaultdict(list)

for pkl_path in PICKLES:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    for comp in data["cluster_compositions"]:
        size = sum(comp.values())
        sizes.append(size)

        class_counts = defaultdict(int)
        for rn, count in comp.items():
            cls = get_class(rn)
            if cls:
                class_counts[cls] += count
        total = sum(class_counts.values())
        if total == 0:
            continue
        for cls in CLASS_COLORS:
            cluster_class_fracs[cls].append(class_counts.get(cls, 0) / total)

sizes = np.array(sizes)
sizes = sizes[sizes >= 2]

mean_fracs = {cls: np.mean(vals) for cls, vals in cluster_class_fracs.items()}
mean_fracs = {
    cls: v for cls, v in sorted(mean_fracs.items(), key=lambda x: x[1]) if v > 0.001
}

# ── Plot: side-by-side, 50/50 widths ─────────────────────────────────────────
fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

# Left: cluster size histogram
ax_hist.hist(
    sizes,
    bins=np.arange(2, sizes.max() + 1),
    color=TEAL,
    alpha=0.5,
    edgecolor=TEAL,
    linewidth=1.2,
    density=True,
)
ax_hist.set_yscale("log")
ax_hist.set_xlim(1.75, sizes.max() + 0.25)
ax_hist.set_xlabel("Cluster size (# metabolites)")
ax_hist.set_ylabel("Density")

# Right: composition bar chart
labels = [CLASS_LABELS[c] for c in mean_fracs]
values = [v * 100 for v in mean_fracs.values()]
colors = [CLASS_COLORS[c] for c in mean_fracs]

bars = ax_bar.barh(
    labels, values, color=colors, edgecolor="white", linewidth=1.5, height=0.7
)

for bar, v in zip(bars, values):
    ax_bar.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.0f}%",
        va="center",
    )

ax_bar.set_xlabel("Mean cluster composition (%)")
ax_bar.set_xlim(0, max(values) * 1.18)
ax_bar.tick_params(axis="y", length=0)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.set_axisbelow(True)

fig.tight_layout()
fig.savefig("figure_3AB.png", bbox_inches="tight", dpi=300)
