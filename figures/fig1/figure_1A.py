import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

met = pd.read_csv("cytosol_composition_metabolites.csv")
info = pd.read_csv("../../processed_data/info.csv")

merged = met.merge(
    info[["resname", "class"]], left_on="Resname", right_on="resname", how="left"
)
by_class = merged.groupby("class")["Concentration (M)"].sum()
mol_pct = (by_class / by_class.sum() * 100).sort_values(ascending=False)

MET_COLORS = {
    "Nucleotides": "#7443BB",
    "Amino_Acids": "#F9B714",
    "Carbohydrates": "#3FB760",
    "Cofactors": "#B7C61E",
    "Lipids": "#05938E",
    "Ions": "#5B81F6",
    "Other_Metabolites": "#CC7FDB",
}
MET_LABELS = {
    "Lipids": "Lipids",
    "Ions": "Ions",
    "Carbohydrates": "Carbohydrates",
    "Nucleotides": "Nucleotides",
    "Amino_Acids": "Amino acids",
    "Other_Metabolites": "Other metabolites",
    "Cofactors": "Cofactors",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

GRAY = "#2a2a2a"
LGRAY = "#888888"

fig, (ax_bar, ax_foot) = plt.subplots(
    2, 1, figsize=(11.0, 1.05), gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0}
)
fig.patch.set_facecolor("white")

# ── Row 1: stacked bar ───────────────────────────────────────────────────────
ax_bar.set_facecolor("white")
min_label_pct = 9.5  # only Lipids (38%) and Ions (22%) get in-bar labels

left = 0.0
small_segs = {}
for cls, pct in mol_pct.items():
    col = MET_COLORS.get(cls, "#AAAAAA")
    ax_bar.barh(
        0, pct, left=left, height=1.0, color=col, edgecolor="white", linewidth=1.0
    )
    if pct >= min_label_pct:
        cx = left + pct / 2
        ax_bar.text(
            cx,
            0.15,
            MET_LABELS.get(cls, cls),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
        ax_bar.text(
            cx,
            -0.18,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            alpha=0.88,
        )
    else:
        small_segs[cls] = pct
    left += pct


ax_bar.set_xlim(0, 100)
ax_bar.set_ylim(-0.5, 0.5)
ax_bar.axis("off")

# ── Row 2: footer — two halves, left=small segs, right=system stats ──────────
ax_foot.set_facecolor("white")
ax_foot.axis("off")
ax_foot.axhline(1.0, color="#DDDDDD", linewidth=0.5)

# Small segments: color swatch + label, spaced manually
x = 0.000
sw = 0.017  # swatch width in axes fraction
sh = 0.50  # swatch height in axes fraction
sy = 0.25  # swatch y bottom

for i, (cls, pct) in enumerate(small_segs.items()):
    col = MET_COLORS.get(cls, "#AAAAAA")
    ax_foot.add_patch(
        mpatches.FancyBboxPatch(
            (x, sy),
            sw,
            sh,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=col,
            transform=ax_foot.transAxes,
            clip_on=False,
        )
    )
    x += sw + 0.008
    label = f"{MET_LABELS.get(cls, cls)} ({pct:.1f}%)"
    ax_foot.text(
        x,
        0.45,
        label,
        transform=ax_foot.transAxes,
        fontsize=10,
        color=GRAY,
        va="center",
        ha="left",
        fontweight="bold",
    )
    x += len(label) * 0.0055 + 0.025  # gap between entries

fig.tight_layout(pad=0)
fig.savefig("figure_1A.png", bbox_inches="tight", transparent=True)
