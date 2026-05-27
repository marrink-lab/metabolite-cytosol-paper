#!/usr/bin/env python3
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("../../mystyle.mplstyle")

STATES = ["protein_adsorbed", "soluble", "clustered"]
COLORS = {"protein_adsorbed": "#8C64C8", "soluble": "#D26496", "clustered": "#FF9664"}

SMOOTH_WINDOW = 10
FIRST_N = 500


def smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(col, kernel, mode="valid") for col in x.T]).T


# ── Load ─────────────────────────────────────────────────────────────────────
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

proportion_values = proportion_values[:FIRST_N]
proportion_errs = proportion_errs[:FIRST_N]

ERR_COLOR = "#A89BB5"  # muted lavender-grey that sits between the purple and pink

fig, ax = plt.subplots(figsize=(12.5, 7.5))

colors = ["#D26496", "#8C64C8", "#FF9664"]

# Create stacked area plot
ax.stackplot(
    proportion_values[:, 0],  # time
    proportion_values[:, 2],  # soluble
    proportion_values[:, 1],  # protein adsorbed
    proportion_values[:, 3],  # clustered soluble
    colors=colors,
    alpha=0.9,
    linewidth=0,
    labels=["Soluble", "Protein-associated", "Clustered"],
)

ax.fill_between(
    proportion_values[:, 0],  # time
    proportion_values.T[2] - proportion_errs.T[2],
    proportion_values.T[2] + proportion_errs.T[2],
    color=ERR_COLOR,
    alpha=1.0,
)

ax.fill_between(
    proportion_values[:, 0],  # time
    proportion_values.T[1:3].sum(axis=0) - proportion_errs.T[1:3].sum(axis=0),
    proportion_values.T[1:3].sum(axis=0) + proportion_errs.T[1:3].sum(axis=0),
    color=ERR_COLOR,
    alpha=1.0,
)


ax.legend(
    fontsize=24,
    ncols=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    bbox_transform=ax.transAxes,
    frameon=False,
)

# Professional styling
ax.set_xlim(proportion_values[0, 0], proportion_values[-1, 0])
ax.set_ylim(0, 100)
ax.set_xlabel("Simulation time (ns)", fontsize=27, fontweight="normal")
ax.set_ylabel("Fraction of metabolites (%)", fontsize=27, fontweight="normal")
ax.tick_params(labelsize=20)
ax.grid(False)

fig.savefig("early_partitioning_evolution.png", dpi=300, bbox_inches="tight")
