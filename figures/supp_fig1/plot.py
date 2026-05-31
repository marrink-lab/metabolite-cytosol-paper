#!/usr/bin/env python3
import glob
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.style.use("../../mystyle.mplstyle")

# ── Configuration ─────────────────────────────────────────────────────────────
T_MAX_NS = 500  # full simulation range to display

# Bottom panel: cluster size
PKL_W_MET = "with_metabolites_6A.pkl"
TIME_STEP_NS = 0.2
SMOOTH_SIGMA_CLUSTER = 5
TRACE_CMAP = "viridis"

# Top panel: partitioning evolution
STATES = ["protein_adsorbed", "soluble", "clustered"]
SMOOTH_WINDOW = 10
STACK_COLORS = [
    "#D26496",
    "#8C64C8",
    "#FF9664",
]  # soluble, protein-associated, clustered
ERR_COLOR = "#A89BB5"


# ── Cluster analysis ──────────────────────────────────────────────────────────
def cluster_size_traces(replicates):
    """
    Return per-replicate (time, smoothed mean-cluster-size) traces.
    """
    times, traces = [], []
    for contacts in replicates:
        ordered = sorted(contacts, key=lambda x: x["frame"])
        means = []
        for graph in ordered:
            G = nx.Graph()
            G.add_nodes_from(n["id"] for n in graph["nodes"])
            G.add_edges_from((l["source"], l["target"]) for l in graph["links"])
            means.append(np.mean([len(c) for c in nx.connected_components(G)]))
        traces.append(gaussian_filter1d(means, sigma=SMOOTH_SIGMA_CLUSTER))
        times.append(np.array([g["frame"] * TIME_STEP_NS for g in ordered]))
    return times, traces


def smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(col, kernel, mode="valid") for col in x.T]).T


# ── Load bottom-panel data ────────────────────────────────────────────────────
with open(PKL_W_MET, "rb") as f:
    data_w = pickle.load(f)

# ── Load top-panel data ───────────────────────────────────────────────────────
results_files = sorted(
    glob.glob("../../analysed_data/with_metabolites/*/cluster_states_*.pkl")
)
results_list = [pickle.load(open(f, "rb"))["results"] for f in results_files]

results = []
for result in results_list:
    analysed_results = np.zeros((len(result), 4))
    for idx, frame_data in enumerate(result):
        fractions = frame_data["fractions"]
        totals = np.array(
            [
                sum(d[k] for d in fractions.values()) / len(fractions.values()) * 100
                for k in STATES
            ]
        )
        analysed_results[idx, 0] = frame_data["time"] / 1000  # ns
        analysed_results[idx, 1:] = totals
    results.append(analysed_results)

# Truncate all replicates to the shortest so np.stack doesn't fail
min_frames = min(len(r) for r in results)
results = [r[:min_frames] for r in results]
proportions = np.stack(results)

proportion_values = smooth(proportions.mean(axis=0), SMOOTH_WINDOW)
proportion_errs = smooth(proportions.std(axis=0), SMOOTH_WINDOW)

# ── Cluster series: one smoothed trace per replicate, full range ──────────────
cluster_times, cluster_traces = cluster_size_traces(data_w["replicates"])

# ── Figure: partitioning (top) + cluster size (bottom), shared time axis ──────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 9), sharex=True, gridspec_kw={"hspace": 0.1}
)

# ── Top panel: partitioning evolution ─────────────────────────────────────────
ax1.stackplot(
    proportion_values[:, 0],  # time
    proportion_values[:, 2],  # soluble
    proportion_values[:, 1],  # protein adsorbed
    proportion_values[:, 3],  # clustered
    colors=STACK_COLORS,
    alpha=0.9,
    linewidth=0,
    labels=["Soluble", "Protein-associated", "Clustered"],
)

ax1.fill_between(
    proportion_values[:, 0],
    proportion_values.T[2] - proportion_errs.T[2],
    proportion_values.T[2] + proportion_errs.T[2],
    color=ERR_COLOR,
    alpha=1.0,
)
ax1.fill_between(
    proportion_values[:, 0],
    proportion_values.T[1:3].sum(axis=0) - proportion_errs.T[1:3].sum(axis=0),
    proportion_values.T[1:3].sum(axis=0) + proportion_errs.T[1:3].sum(axis=0),
    color=ERR_COLOR,
    alpha=1.0,
)

ax1.legend(
    fontsize=16,
    ncols=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    bbox_transform=ax1.transAxes,
    frameon=False,
)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Fraction of metabolites (%)", fontsize=20)

# ── Bottom panel: individual replicate cluster-size traces ────────────────────
trace_colors = plt.get_cmap(TRACE_CMAP)(np.linspace(0.15, 0.85, len(cluster_traces)))
ymax = 0.0
for i, (ti, tr) in enumerate(zip(cluster_times, cluster_traces)):
    ax2.plot(
        ti,
        tr,
        color=trace_colors[i],
        lw=1.5,
        alpha=0.9,
        solid_capstyle="round",
        label=f"Replicate {i + 1}",
    )
    ymax = max(ymax, tr[ti <= T_MAX_NS].max())

ax2.set_ylabel("Mean cluster size\n(no. of proteins)", fontsize=20)
ax2.set_ylim(0, ymax * 1.1)
ax2.legend(loc="upper left", frameon=False, fontsize=13, ncols=2)

# ── Shared x-axis: full simulation range ──────────────────────────────────────
ax2.set_xlim(0, T_MAX_NS)
ax2.set_xlabel("Simulation time (ns)", fontsize=20)

for ndx, ax in enumerate((ax1, ax2)):
    ax.tick_params(labelsize=15)
    if ndx == 0:
        ax.grid(False)
    else:
        ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig(
    "cluster_and_partitioning.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
