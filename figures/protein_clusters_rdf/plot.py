#!/usr/bin/env python3
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.style.use("../../mystyle.mplstyle")

# ── Configuration ─────────────────────────────────────────────────────────────
PKL_W_MET = "with_metabolites_6A.pkl"
PKL_WO_MET = "no_metabolites_6A.pkl"
RDF_PKL = "protein_rdf.pkl"

TIME_STEP_NS = 0.2
SMOOTH_SIGMA_CLUSTER = 5
SMOOTH_SIGMA_RDF = 5

COLOR_W = "#05938E"
COLOR_WO = "#666666"


# ── Cluster analysis ──────────────────────────────────────────────────────────
def mean_cluster_size_per_replicate(replicates):
    all_means = []
    for contacts in replicates:
        means = []
        for graph in sorted(contacts, key=lambda x: x["frame"]):
            G = nx.Graph()
            G.add_nodes_from(n["id"] for n in graph["nodes"])
            G.add_edges_from((l["source"], l["target"]) for l in graph["links"])
            means.append(np.mean([len(c) for c in nx.connected_components(G)]))
        all_means.append(gaussian_filter1d(means, sigma=SMOOTH_SIGMA_CLUSTER))

    all_means = np.array(all_means)
    times = np.array(
        [
            g["frame"] * TIME_STEP_NS
            for g in sorted(replicates[0], key=lambda x: x["frame"])
        ]
    )
    return times, all_means.mean(axis=0), all_means.std(axis=0)


# ── Load ──────────────────────────────────────────────────────────────────────
with open(PKL_W_MET, "rb") as f:
    data_w = pickle.load(f)
with open(PKL_WO_MET, "rb") as f:
    data_wo = pickle.load(f)
with open(RDF_PKL, "rb") as f:
    rdf = pickle.load(f)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25}
)

# ── Panel a: mean cluster size ────────────────────────────────────────────────
for data, color, label in (
    (data_w, COLOR_W, "With metabolites"),
    (data_wo, COLOR_WO, "Without metabolites"),
):
    t, mean, std = mean_cluster_size_per_replicate(data["replicates"])
    ax1.plot(t, mean, color=color, lw=2.0, solid_capstyle="round", label=label)
    ax1.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

ax1.set_xlabel("Time (ns)", fontsize=15)
ax1.set_ylabel("Mean cluster size (no. of proteins)", fontsize=15)
ax1.set_xlim(left=0, right=500)
ax1.legend(loc="upper left", frameon=False, fontsize=10.5)

# ── Panel b: RDF ──────────────────────────────────────────────────────────────
for key, color, label in (
    ("with_metabolites", COLOR_W, "With metabolites"),
    ("without_metabolites", COLOR_WO, "Without metabolites"),
):
    r = rdf[key]["r"] / 10
    smoothed = np.array(
        [gaussian_filter1d(gr, sigma=SMOOTH_SIGMA_RDF) for gr in rdf[key]["gr_reps"]]
    )
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)
    ax2.plot(r, mean, color=color, lw=2.0, solid_capstyle="round", label=label)
    ax2.fill_between(r, mean - std, mean + std, color=color, alpha=0.2)

ax2.axhline(1.0, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
ax2.set_xlabel("r (nm)", fontsize=15)
ax2.set_ylabel("g(r)", fontsize=15)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1.75)
ax2.legend(loc="upper left", frameon=False, fontsize=10.5)

# ── Shared style ──────────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig("clustering_and_rdf.png", dpi=300, bbox_inches="tight", facecolor="white")
