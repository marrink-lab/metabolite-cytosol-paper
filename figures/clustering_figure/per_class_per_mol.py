#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from collections import defaultdict
import requests
import io
import pandas as pd
from uncertainties import ufloat

plt.style.use("../../mystyle.mplstyle")


STATES = ["protein_adsorbed", "soluble", "clustered"]
COLORS = {'protein_adsorbed': '#8C64C8', 
          'soluble': '#D26496', 
          'clustered': '#FF9664'}


def compute_time_averaged_fractions(results):
    """
    time averaged fractions for a single replica
    """
    accumulator = defaultdict(lambda: defaultdict(list))

    start = int(len(results)/2)
    for frame in results[start:]:
        for mol, states in frame['fractions'].items():
            for state, val in states.items():
                accumulator[mol][state].append(val)

    avg_frac = {
        mol: {state: np.mean(vals) for state, vals in states.items()}
        for mol, states in accumulator.items()
    }

    return avg_frac


def average_time_averaged_fractions_across_replicas(results_list):
    """
    Parameters
    ----------
    results_list : list
        Each element is a per-frame results object

    Returns
    -------
    avg_frac_mean : dict
        avg_frac_mean[molecule][state] = replica-averaged mean
    avg_frac_std : dict
        avg_frac_std[molecule][state] = std across replicas
    """

    # First: compute time-averaged fractions per replica
    replica_avgs = []
    for results in results_list:
        replica_avgs.append(
            compute_time_averaged_fractions(results)
        )

    # Collect across replicas
    accumulator = defaultdict(lambda: defaultdict(list))

    for avg_frac in replica_avgs:
        for mol, states in avg_frac.items():
            for state, val in states.items():
                accumulator[mol][state].append(val)

    # Compute mean and std across replicas
    avg_frac_mean = {}
    avg_frac_std = {}

    for mol, states in accumulator.items():
        avg_frac_mean[mol] = {}
        avg_frac_std[mol] = {}
        for state, vals in states.items():
            avg_frac_mean[mol][state] = np.mean(vals)
            avg_frac_std[mol][state] = np.std(vals)

    return avg_frac_mean, avg_frac_std


def plot_time_averaged_by_class(avg_frac, std_frac, mol_to_class):
    """
    Produces one subplot per metabolite class.
    Each subplot contains three box plots:
        protein_adsorbed / soluble / clustered
    """

    states = ["protein_adsorbed", "soluble", "clustered"]
    state_labels = ["Protein adsorbed", "Soluble", "Clustered"]
    colors = ['#8C64C8','#D26496','#FF9664']
    cols=dict(zip(states, colors))
    
    # Organise data: class → state → list of values
    class_data = defaultdict(lambda: defaultdict(list))

    for mol, states_dict in avg_frac.items():
        if mol not in mol_to_class:
            continue
        cls = mol_to_class[mol]
        for state in states:
            class_data[cls][state].append(states_dict[state])

    # Organise data: class → state → list of values
    class_data_std = defaultdict(lambda: defaultdict(list))

    for mol, states_dict in std_frac.items():
        if mol not in mol_to_class:
            continue
        cls = mol_to_class[mol]
        for state in states:
            class_data_std[cls][state].append(states_dict[state])

    classes = sorted(class_data.keys())
    n_classes = len(classes)

    fig, axes = plt.subplots(
        2,4,
        figsize=(15,7.5),
        sharey=True
    )

    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes.flatten(), classes):
        data = [class_data[cls][state] for state in states]

        ax.boxplot(
            data,
            widths=0.6,
            showfliers=False,
            boxprops={'lw': 2,
                      'color': '#262626'},
            medianprops={'lw':2,
                         'c': '#262626'},
            capprops={'lw':2,
                      'c': '#262626'},
            whiskerprops={'lw':2,
                          'c': '#262626'},
            flierprops = {'marker': ''},
            # patch_artist=True,
            zorder=10           
        )

        ax.set_title(' '.join(cls.split('_')),
                     fontsize=20)
        ax.set_xticks([])
        ax.set_ylim(0, 1)

        # Optional: show individual metabolite means as scatter
        for i, state in enumerate(states, start=1):
            y = class_data[cls][state]
            yerr = class_data_std[cls][state]
            x = np.random.normal(i, 0.04, size=len(y))
            ax.errorbar(x, y, 
                        yerr=yerr,
                        label=state_labels[i-1], #' '.join([j for j in state_labels[i-1].split('_')]),
                        #alpha=0.6,
                        c=cols[state],
                        markeredgecolor='#262626',
                        markeredgewidth=.25,
                        ls='',
                        marker='.',
                        markersize=25,
                        )

    for ax in axes[:,0]:
        ax.set_ylabel('Fraction (%)',
                      fontsize=20)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],
                      ['0', '20', '40', '60', '80', '100'])
            
    axes[-1][-2].legend(#handles,
                         #labels,
                         loc = 'center',
                         handletextpad=0.1,
                         bbox_transform = axes[-1][-1].transAxes,
                         bbox_to_anchor=(0.5, 0.5),
                         fontsize = 20,
                         markerscale=2)
    fig.delaxes(axes[-1][-1])
    fig.savefig('clustering_per_class_per_mol.png', 
                dpi = 500,
                bbox_inches='tight')


def export_data(mol_to_class, avg_frac_mean, avg_frac_std):
        
    data_out = defaultdict(lambda: defaultdict(dict))
    states = ["protein_adsorbed", "soluble", "clustered"]
    
    for mol, states_dict in avg_frac_mean.items():
        if mol not in mol_to_class:
            continue
        cls = mol_to_class[mol]
        vals_dict = {}
        for state in states:
            vals_dict[state] = ufloat(np.round(states_dict[state], 2),
                                      np.round(avg_frac_std[mol][state], 2))
        data_out[cls][mol] = vals_dict
            
    dfs = {
        key: pd.DataFrame.from_dict(value, orient="index")
        for key, value in data_out.items()
    }
            
    with pd.ExcelWriter("per_class_per_mol.xlsx", engine="xlsxwriter") as writer:
        for sheet_name, df in dfs.items():
            df = dfs[sheet_name].sort_index()
            df.to_excel(writer, sheet_name=sheet_name)

# ── Load ─────────────────────────────────────────────────────────────────────
results_files = sorted(glob.glob("../../analysed_data/*/cluster_states_*.pkl"))
results_list  = [pickle.load(open(f, 'rb'))['results'] for f in results_files]


# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(io.StringIO(s.decode('utf-8')),
                          usecols=[
                              'resname',
                              'class',
                              ],
                          index_col='resname')

avg_frac_mean, avg_frac_std = average_time_averaged_fractions_across_replicas(results_list)
mol_to_class = metabolites.to_dict()['class']

plot_time_averaged_by_class(avg_frac_mean,
                            avg_frac_std,
                            mol_to_class
                            )

export_data(mol_to_class, avg_frac_mean, avg_frac_std)


