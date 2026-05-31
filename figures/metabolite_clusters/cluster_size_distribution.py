import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("../../mystyle.mplstyle")

PICKLES = [
    "cluster_compositions_rep0.pkl",
    "cluster_compositions_rep1.pkl",
    "cluster_compositions_rep2.pkl",
]

TEAL = "#05938E"

sizes = []
for pkl_path in PICKLES:
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    for comp in data["cluster_compositions"]:
        sizes.append(sum(comp.values()))
sizes = np.array(sizes)
sizes = sizes[sizes >= 2]

fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(
    sizes,
    bins=np.arange(2, sizes.max() + 1),
    color=TEAL,
    alpha=0.5,
    edgecolor=TEAL,
    linewidth=1.2,
    density=True,
)
ax.set_yscale("log")
ax.set_xlim(1.75, sizes.max() + 0.25)
ax.set_xlabel("Cluster size (# metabolites)")
ax.set_ylabel("Density")
ax.tick_params()

fig.savefig("fig3A.png", bbox_inches="tight", dpi=300)
