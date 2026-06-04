import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    "Nucleotides": "Nucleotides",
    "Amino_Acids": "Amino acids",
    "Carbohydrates": "Carbohydrates",
    "Cofactors": "Cofactors",
    "Fatty_Acids": "Fatty acids",
    "Ions": "Ions",
    "Other_Metabolites": "Other",
}
CLASSES = list(CLASS_COLORS)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
n_classes = len(CLASSES)

VMAX = 1.0
TEXT_COLOR = "#444444"

mol_to_class = dict(
    zip(*pd.read_csv("../../processed_data/info.csv")[["resname", "class"]].values.T)
)

M = np.zeros((n_classes, n_classes))
T = np.zeros(n_classes)
unmapped = set()

for pkl_path in PICKLES:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    for comp in data["cluster_compositions"]:
        c = np.zeros(n_classes)
        for resname, n in comp.items():
            cls = mol_to_class.get(resname)
            if cls is None:
                unmapped.add(resname)
                continue
            c[class_to_idx[cls]] += n
        T += c
        M += np.outer(c, c)
        M -= np.diag(c)

if unmapped:
    print(
        f"Warning: {len(unmapped)} resnames missing from molecules_list.csv: {sorted(unmapped)}"
    )
assert T.sum() > 0, "no clustered metabolites matched any class"

total_pairs = M.sum()
f = T / T.sum()
E = np.outer(f, f) * total_pairs

with np.errstate(divide="ignore", invalid="ignore"):
    enrichment = np.log2(M / E)
enrichment = np.where(np.isfinite(enrichment), enrichment, np.nan)

# --- Plot ---
plt.style.use("../../mystyle.mplstyle")

fig, ax = plt.subplots(figsize=(4.8, 3.8))

im = ax.imshow(
    enrichment, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, interpolation="nearest"
)
ax.grid(False)
ax.set_xticks([])

# Top colour strip identifies columns; y-tick labels identify rows
strip_rgb = np.array([to_rgb(CLASS_COLORS[c]) for c in CLASSES])
divider = make_axes_locatable(ax)
ax_top = divider.append_axes("top", size=0.13, pad=0.04)
ax_top.imshow(
    strip_rgb[np.newaxis, :, :],
    aspect="auto",
    extent=(-0.5, n_classes - 0.5, 0, 1),
    interpolation="nearest",
)
ax_top.set_xticks([])
ax_top.set_yticks([])
for spine in ax_top.spines.values():
    spine.set_visible(False)
ax_top.grid(False)

ax_left = divider.append_axes("left", size=0.18, pad=0.04)

ax_left.imshow(
    strip_rgb[:, np.newaxis, :],
    aspect="auto",
    extent=(0, 1, n_classes - 0.5, -0.5),
    interpolation="nearest",
)

ax_left.set_yticks(range(n_classes))
ax_left.set_yticklabels([CLASS_LABELS[c] for c in CLASSES], color=TEXT_COLOR)
ax_left.set_xticks([])
ax_left.tick_params(axis="y", length=0, pad=4)
ax_left.grid(False)
ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.grid(False)
for a in (ax_top, ax_left):
    for spine in a.spines.values():
        spine.set_visible(False)

# Horizontal colorbar below
cax = divider.append_axes("bottom", size=0.13, pad=0.15)
cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_ticklabels(["½×", "0.7×", "1×", "1.4×", "2×"])
cbar.ax.tick_params(colors=TEXT_COLOR, length=0, pad=2)
cbar.outline.set_visible(False)
cbar.set_label(
    "Pairwise enrichment in clusters", color=TEXT_COLOR, labelpad=4, fontsize=9
)

fig.savefig("supp_fig5.png", bbox_inches="tight", dpi=300)
