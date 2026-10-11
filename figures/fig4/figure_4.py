#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import brunnermunzel, gaussian_kde, mannwhitneyu
from tqdm import tqdm

def auc_effect_size(x1, x2):
    """
    Common-language effect size, AUC = U / (n1 * n2) = P(X1 > X2) + 0.5 * P(X1 == X2).

    0.5 means no tendency for either group to rank higher; < 0.5 means x1 tends to be lower.
    """
    U, _ = mannwhitneyu(x1, x2, alternative="two-sided")
    return U / (len(x1) * len(x2))


def bootstrap_auc_ci(x1, x2, n_iter=10000, ci=95, random_state=None):
    """
    Percentile bootstrap CI on the AUC, resampling proteins (with replacement)
    within each group.

    Returns:
        aucs : array of bootstrapped AUC values
        lo, hi : lower and upper bounds of the CI
    """
    rng = np.random.default_rng(random_state)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    aucs = np.empty(n_iter)
    for i in tqdm(range(n_iter)):
        s1 = rng.choice(x1, size=len(x1), replace=True)
        s2 = rng.choice(x2, size=len(x2), replace=True)
        aucs[i] = auc_effect_size(s1, s2)

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(aucs, [alpha, 100 - alpha])
    return aucs, lo, hi


"""
Some high level things to set up everything else
"""

base = "../../.."
seed = 2025
rng = np.random.default_rng(seed)

plt.style.use(f"{base}/mystyle.mplstyle")

colors = {
    "Ions": "#337AEA",  # bright cobalt
    "Lipids": "#05938E",  # jade teal
    "Nucleotides": "#944CE5",  # bright violet
    "Cofactors": "#B7C61E",  # bright olive
    "Amino_Acids": "#F9B714",  # bright warm yellow
    "Other_Metabolites": "#CC7FDB",  # bright mauve
    "Carbohydrates": "#3FB760",  # bright fern
}

cl_labels = [
    "Ions",
    "Fatty\nacid",
    "Nucleotides",
    "Cofactors",
    "Amino\nacids",
    "Other",
    "Carbohyd.",
]
classes = list(colors.keys())

fig, (ax0, ax1, ax2) = plt.subplots(
    3,
    1,
    figsize=(15, 30),
)

"""
Diffusion
"""

diffusion_dict = pickle.load(
    open(f"{base}/processed_data/diffusion/diffusion_coefficients.pkl", "rb")
)

data_in = []
for cl in colors.keys():
    data_in.append([j for i, j in diffusion_dict["results"][cl].items()])

violin = [[j[0] for j in i] for i in data_in]
parts = ax0.violinplot(
    violin,
    positions=np.arange(len(violin)),
    showmedians=True,
)

for i, j in enumerate(data_in):
    ax0.errorbar(
        rng.normal(i, 0.05, size=len(j)),
        [k[0] for k in j],
        yerr=[k[1] for k in j],
        marker=".",
        markersize=30,
        markeredgewidth=1,
        markeredgecolor="#262626",
        ls="none",
        color=list(colors.values())[i],
    )

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(colors.values())[i])
    pc.set_linewidth(2)
    pc.set_zorder(10)
    pc.set_alpha(0.3)
    pc.set_edgecolor("#262626")

for i, j in parts.items():
    if i == "bodies":
        continue
    else:
        parts[i].set_colors("#262626")
        parts[i].set_zorder(10)
        parts[i].set_linewidth(5)

"""
lifetimes
"""

diffusion = pd.read_csv(
    f"{base}/processed_data/lifetimes/residence_exponents.csv"
).sort_values("exponent", ascending=False)
cols = [colors[i] for i in diffusion["class"]]

ax1.bar(
    cl_labels,
    diffusion["exponent"],
    color=cols,
    edgecolor="white",
    linewidth=2.0,
    yerr=diffusion["exponent_err"],
)


#####################
### LOAD THE DATA ###
#####################

datasets = [
    pickle.load(
        open(f"{base}/analysed_data/with_metabolites/replica_{i}/binding_{i}.pkl", "rb")
    )
    for i in range(1, 4)
]

surface_areas = pickle.load(open(f"{base}/figures/_revision/protein_sasa.pkl", "rb"))

proteomics = pd.read_csv(
    f"{base}/figures/_revision/proteomics_annotated.csv", index_col="Locus tag"
)

# # get metabolite classes from the master database spreadsheet on the M3-metabolome repo
# metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
# s = requests.get(metabolites_url).content

# metabolites = pd.read_csv(
#     io.StringIO(s.decode("utf-8")),
#     usecols=["Metabolite name", "resname", "class"],
#     index_col="Metabolite name",
# )
metabolites = pd.read_csv('metabolites.csv')

metabolite_classes = metabolites.to_dict()["class"]

# for each replica, for each protein, get the number of ATP contacts normalised by
# the number of copies of the protein and its solvent accessible surface area.
# we end up with a dictionary of {protein: [normalised_count_0, normalised_count_1... normalised_count_n]}
ATP_results = defaultdict(list)
for idx, d in enumerate(datasets):
    ATP_vals = {
        protein: {
            "ATP_count": value.get("ATPH", {}).get("count", 0),
            "protein_count": d["n_proteins"][protein],
            "protein_SASA": surface_areas.get(protein),
        }
        for protein, value in d["protein_results"].items()
    }
    for protein, values in ATP_vals.items():
        ATP_results[protein].append(
            values["ATP_count"] / (values["protein_count"] * values["protein_SASA"])
        )


# from the proteomics, make a dictionary indicating whether the each protein is a known binder of ATP or not.
# compare explicitly against True so that a missing annotation (NaN, which is truthy) is not counted as a binder
ATP_mask = {}
for protein in proteomics.index:
    ligs = proteomics.loc[protein]["atp_binding"]
    protein_renamed = protein[-3:] + "_monomer"
    ATP_mask[protein_renamed] = bool(ligs == True)

# combine the results with the mask.
# each protein is summarised by its mean over replicas, with the standard error of
# that mean (ddof=1) used only for the error bars in the plot
arr = []
names = []
mask = []
for key, val in ATP_results.items():
    if key in ATP_mask.keys():
        mask.append(ATP_mask[key])
        sem = np.std(val, ddof=1) / np.sqrt(len(val)) if len(val) > 1 else 0.0
        arr.append(np.array([np.mean(val), sem]))
        names.append(key)
arr = np.array(arr)
mask = np.array(mask)

# set things up for statistical testing
binding_values = arr.T[0]
binding_errors = arr.T[1]

known_binders = binding_values[mask]
non_binders = binding_values[~mask]
known_binders_err = binding_errors[mask]
non_binders_err = binding_errors[~mask]


violin_width = 0.5  # matches matplotlib's default `widths` arg
violin1 = ax2.violinplot(
    [known_binders, non_binders], positions=[0, 1], widths=violin_width
)

for i, pc in enumerate(violin1["bodies"]):
    pc.set_linewidth(2)
    pc.set_zorder(10)
    pc.set_edgecolor("#262626")


# Overlay the scatter plot for individual data points
for pos, group_vals, group_errs in zip(
    [0, 1], [known_binders, non_binders], [known_binders_err, non_binders_err]
):
    kde = gaussian_kde(group_vals)  # same density estimator violinplot uses
    dens = kde(group_vals)  # density at each point's own y-value
    dens_norm = dens / dens.max()  # rescale to [0, 1] (1 = violin's widest spot)

    # Add small horizontal jitter so points don't overlap completely
    half_width = violin_width / 3
    jitter = rng.uniform(-1, 1, size=len(group_vals)) * dens_norm * half_width
    ax2.errorbar(
        pos + jitter,
        group_vals,
        yerr=group_errs,
        color="black",
        alpha=0.3,
        lw=0.5,
        marker=".",
        markersize=30,
        markeredgewidth=1,
        markeredgecolor="#262626",
        ls="none",
        zorder=1,
    )


#########################
### STATISTICAL TESTS ###
#########################

# the unit of observation is the protein (mean over replicas). replicate noise is
# already part of the between-protein spread, so no extra perturbation is applied.
U_mw, p_mw = mannwhitneyu(known_binders, non_binders, alternative="two-sided")

# Brunner-Munzel tests the same null, P(X > Y) = 0.5, without assuming equal variances / shapes
bm_stat, p_bm = brunnermunzel(known_binders, non_binders, alternative="two-sided")

auc = U_mw / (len(known_binders) * len(non_binders))
rank_biserial = 2 * auc - 1

boot_iters = 100000
_, auc_lo, auc_hi = bootstrap_auc_ci(
    known_binders, non_binders, n_iter=boot_iters, ci=95, random_state=seed
)

stats_report = []

stats_report.append(f"No. known ATP binders: {len(known_binders)}")
stats_report.append(
    f"Known ATP binders mean / median: {known_binders.mean():.3g} / {np.median(known_binders):.3g}\n"
)

stats_report.append(f"No. ATP non-binders: {len(non_binders)}")
stats_report.append(
    f"ATP non-binders mean / median: {non_binders.mean():.3g} / {np.median(non_binders):.3g}\n"
)

stats_report.append("Mann-Whitney U (two-sided)")
stats_report.append(f"U = {U_mw:.1f}, p = {p_mw:.3g}\n")

stats_report.append("Brunner-Munzel (two-sided)")
stats_report.append(f"W = {bm_stat:.3f}, p = {p_bm:.3g}\n")

stats_report.append("Effect size")
stats_report.append(f"AUC = P(binder > non-binder) = {auc:.3f}")
stats_report.append(
    f"95% bootstrap CI on AUC ({boot_iters} resamples of proteins): [{auc_lo:.3f}, {auc_hi:.3f}]"
)
stats_report.append(f"Rank-biserial correlation = {rank_biserial:.3f}")

with open("stats_report.txt", "w") as f:
    f.writelines("\n".join(stats_report))

print("\n".join(stats_report))

# ax2.text(
#     0.98,
#     0.98,
#     f"Mann-Whitney p = {p_mw:.2g}\n"
#     f"Brunner-Munzel p = {p_bm:.2g}\n"
#     f"AUC = {auc:.2f} [{auc_lo:.2f}, {auc_hi:.2f}]",
#     transform=ax2.transAxes,
#     ha="right",
#     va="top",
#     fontsize=25,
# )


"""
plot aesthetics
"""
fig.subplots_adjust(hspace=0.1)


ax0.set_yscale("log")
ax0.set_ylabel("Diffusion (cm$^2$/s)", fontsize=40)


ax0.tick_params(labelsize=20)

ax0.set_xticks(np.arange(len(diffusion)), [""] * len(diffusion))

for i, j in enumerate(cl_labels):
    ax1.text(i, 0.1, j, fontsize=22.5, ma="center", va="center", ha="center")
ax1.set_xticks(np.arange(len(diffusion)), [""] * len(diffusion))

ax1.tick_params(axis="y", labelsize=20, labeltop=True, labelbottom=False)

ax1.set_ylabel("Association lifetime\ndistribution, power\nlaw exponent", fontsize=40)

ax2.set_xticks([0, 1], ["Known binders", "Non binders"])
ax2.set_ylabel(
    "ATP contacts\nper protein copy\nper Å$^2$ SASA", fontsize=40
)
ax2.tick_params(labelsize=20)

fig.savefig("figure_4.png", dpi=500, bbox_inches="tight")
