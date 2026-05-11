#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle


plt.style.use("../../mystyle.mplstyle")

STATES = ["protein_adsorbed", "soluble", "clustered"]
COLORS = {'protein_adsorbed': '#8C64C8', 
          'soluble': '#D26496', 
          'clustered': '#FF9664'}

# ── Load ─────────────────────────────────────────────────────────────────────
results_files = sorted(glob.glob("../../analysed_data/with_metabolites/*/cluster_states_*.pkl"))
results_list  = [pickle.load(open(f, 'rb'))['results'] for f in results_files]

results = []
for result in results_list:
    analysed_results = np.zeros((len(result), 4))
    for idx, frame_data in enumerate(result):
        fractions = frame_data['fractions']
        totals = np.array([
            sum(d[k] for d in fractions.values()) / len(fractions.values())*100
            for k in STATES
        ])
        
        analysed_results[idx, 0] = frame_data['time'] / 1000 # ns
        analysed_results[idx, 1:] = totals
    results.append(analysed_results)

# this line because one didn't quite run to the last frame
results = [i[:250] for i in results]
proportions = np.stack(results)

proportion_values = proportions.mean(axis=0)
proportion_errs = proportions.std(axis=0)


fig, ax = plt.subplots(figsize=(10,7.5))

colors = ['#D26496',
          '#8C64C8',
          '#FF9664']

# Create stacked area plot
ax.stackplot(proportion_values[:, 0],  # time
             proportion_values[:, 2],  # soluble
             proportion_values[:, 1],  # protein adsorbed
             proportion_values[:, 3],  # clustered soluble
             colors=colors,
             alpha=0.9,
             linewidth=0,
             labels=['Soluble', 'Protein-associated', 'Clustered'])

ax.fill_between(proportion_values[:, 0],  # time
                proportion_values.T[2] - proportion_errs.T[2],
                proportion_values.T[2] + proportion_errs.T[2],
                color='#262626')

ax.fill_between(proportion_values[:, 0],  # time
                proportion_values.T[1:3].sum(axis=0) - 
                proportion_errs.T[1:3].sum(axis=0),
                proportion_values.T[1:3].sum(axis=0) + 
                proportion_errs.T[1:3].sum(axis=0),
                color='#262626')


ax.legend(fontsize = 30,
          bbox_to_anchor=(1.7,0.5),
          bbox_transform=ax.transAxes
          )

# Professional styling
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('Simulation time (ns)', fontsize=30, fontweight='normal')
ax.set_ylabel('Fraction of metabolites (%)', fontsize=30, fontweight='normal')
ax.tick_params(labelsize=20)

fig.savefig('early_partitioning_evolution.png',
            dpi = 300,
            bbox_inches='tight')


