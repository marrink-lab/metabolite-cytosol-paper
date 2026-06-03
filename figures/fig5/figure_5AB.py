#!/usr/bin/env python3
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.style.use("../../mystyle.mplstyle")

# ── Configuration ─────────────────────────────────────────────────────────────
CONTACTS_PKL = "protein_contacts.pkl"
RDF_PKL = "protein_rdf.pkl"

SMOOTH_SIGMA_CONTACTS = 5
SMOOTH_SIGMA_RDF = 5

COLOR_W = "#05938E"
COLOR_WO = "#666666"


# ── Contact analysis ──────────────────────────────────────────────────────────
def contacts_mean_std(reps):
    n = min(len(r["n_contacts"]) for r in reps)
    t = reps[0]["time_ns"][:n]
    stack = np.vstack(
        [
            gaussian_filter1d(r["n_contacts"][:n], sigma=SMOOTH_SIGMA_CONTACTS)
            for r in reps
        ]
    )
    return t, stack.mean(axis=0), stack.std(axis=0)


# ── Load ──────────────────────────────────────────────────────────────────────
with open(CONTACTS_PKL, "rb") as f:
    contacts = pickle.load(f)
with open(RDF_PKL, "rb") as f:
    rdf = pickle.load(f)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25}
)

# ── Panel a: direct protein-protein contacts ──────────────────────────────────
for key, color, label in (
    ("with_metabolites", COLOR_W, "With metabolites"),
    ("no_metabolites", COLOR_WO, "Without metabolites"),
):
    t, mean, std = contacts_mean_std(contacts["systems"][key])
    ax1.plot(t, mean, color=color, lw=2.0, solid_capstyle="round", label=label)
    ax1.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

ax1.set_xlabel("Time (ns)", fontsize=15)
ax1.set_ylabel("Direct protein-protein contacts", fontsize=15)
ax1.set_xlim(left=0, right=500)
ax1.set_ylim(bottom=1000)
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
ax2.set_ylim(0, 1.80)
ax2.legend(loc="upper left", frameon=False, fontsize=10.5)

# ── Shared style ──────────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig("figure_5AB.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
