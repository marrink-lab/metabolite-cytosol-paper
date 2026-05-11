#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from collections import defaultdict
import io
import requests

plt.style.use("../../mystyle.mplstyle")

STATES = ["protein_adsorbed", "soluble", "clustered"]
COLORS = {'protein_adsorbed': '#8C64C8', 
          'soluble': '#D26496', 
          'clustered': '#FF9664'}
LABELS = {'protein_adsorbed': 'Protein-adsorbed', 
          'soluble': 'Soluble', 
          'clustered': 'Clustered'}
CLASS_LABELS = {
    'Amino_Acids': 'Amino acids', 
    'Carbohydrates': 'Carbohyd.', 
    'Cofactors': 'Cofactors',
    'Ions': 'Ions', 
    'Lipids': 'Fatty acid', 
    'Nucleotides': 'Nucleotides', 
    'Other_Metabolites': 'Other',
}

# ── Load ──────────────────────────────────────────────────────────────────────
results_files = sorted(glob.glob("../../analysed_data/with_metabolites/*/cluster_states_*.pkl"))
results_list  = [pickle.load(open(f, 'rb'))['results'] for f in results_files]

# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(io.StringIO(s.decode('utf-8')),
                          usecols=['resname',
                                   'class'],
                          index_col='resname')

mol_to_class = metabolites.to_dict()['class']

# ── Time-average fractions (last 50% of trajectory per replicate) ─────────────
def time_avg_fractions(results):
    acc = defaultdict(lambda: defaultdict(list))
    for frame in results[len(results) // 2:]:
        for mol, states in frame['fractions'].items():
            for state, val in states.items():
                acc[mol][state].append(val)
    return {mol: {s: np.mean(v) for s, v in states.items()} for mol, states in acc.items()}

rep_avgs = [time_avg_fractions(r) for r in results_list]

acc = defaultdict(lambda: defaultdict(list))
for rep in rep_avgs:
    for mol, states in rep.items():
        for state, val in states.items():
            acc[mol][state].append(val)

avg_frac = {mol: {s: np.mean(v) for s, v in states.items()} for mol, states in acc.items()}

# ── Aggregate to class level ───────────────────────────────────────────────────
class_data = defaultdict(lambda: defaultdict(list))
for mol, sd in avg_frac.items():
    if mol not in mol_to_class:
        continue
    for state in STATES:
        class_data[mol_to_class[mol]][state].append(sd[state])

class_means = {c: {s: np.mean(v) for s, v in sd.items()} for c, sd in class_data.items()}
classes     = sorted(class_means, key=lambda c: class_means[c]['protein_adsorbed'], reverse=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.2))
fig.patch.set_facecolor('white')

for i, cls in enumerate(classes):
    left = 0.0
    for state in STATES:
        mean = class_means[cls][state]
        ax.barh(i, mean, left=left, height=0.70,
                color=COLORS[state], edgecolor='white', linewidth=0.6)
        if mean > 0.07:
            ax.text(left + mean / 2, i, f'{mean*100:.0f}%',
                    ha='center', va='center', fontsize=7.5, color='white')
        left += mean

ax.set_yticks(range(len(classes)))
ax.set_yticklabels([CLASS_LABELS.get(c, c) for c in classes], fontsize=9)
ax.tick_params(axis='y', pad=5)
ax.set_xlabel('Population fraction (%)', fontsize=12)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}'))

fig.tight_layout(pad=0.5)
fig.savefig('partitioning.png', dpi=300, bbox_inches='tight')
