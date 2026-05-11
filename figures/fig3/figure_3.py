#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import mannwhitneyu

import io
import requests

# from uncertainties.unumpy import uarray
from uncertainties import ufloat, nominal_value
from MDAnalysis.units import constants

def bootstrap_mannwhitney(
    x1, err1, x2, err2,
    n_iter=10000,
    alternative="two-sided",
    random_state=None
):
    """
    Bootstrap Mann–Whitney U test accounting for measurement errors.

    Returns:
        U_vals : array of U statistics
        p_vals : array of p-values
    """
    rng = np.random.default_rng(random_state)

    U_vals = np.empty(n_iter)
    p_vals = np.empty(n_iter)

    for i in tqdm(range(n_iter)):
        # Resample each point according to its measurement error
        sample1 = rng.normal(loc=x1, scale=err1)
        sample2 = rng.normal(loc=x2, scale=err2)

        U, p = mannwhitneyu(sample1, sample2, alternative=alternative)
        U_vals[i] = U
        p_vals[i] = p

    return U_vals, p_vals



"""
Some high level things to set up everything else
"""

base = '../..'

plt.style.use(f"{base}/mystyle.mplstyle")

colors = {
    'Ions':              '#337AEA',  # bright cobalt
    'Lipids':            '#05938E',  # jade teal
    'Nucleotides':       '#944CE5',  # bright violet
    'Cofactors':         '#B7C61E',  # bright olive
    'Amino_Acids':       '#F9B714',  # bright warm yellow
    'Other_Metabolites': '#CC7FDB',  # bright mauve
    'Carbohydrates':     '#3FB760',  # bright fern
}

cl_labels = [
    'Ions',
    'Lipids',
    'Nucleotides',
    'Cofactors',
    'Amino\nacids',
    'Other',
    'Carbohyd.',
    ]
classes = list(colors.keys())

fig, (ax0,ax1, ax2) = plt.subplots(3,1,
                              # sharex=True,
                              figsize=(15,30)
                              )

proteomics = pd.read_csv(f'{base}/processed_data/proteomics_annotated.csv',
                         index_col='Locus tag')

"""
Diffusion
"""

diffusion_dict = pickle.load(open(f'{base}/processed_data/diffusion/diffusion_coefficients.pkl', 'rb'))

data_in = []
for cl in colors.keys():
    data_in.append([j[0] for i,j in diffusion_dict['results'][cl].items()])

parts = ax0.violinplot(data_in,
                       positions = np.arange(len(data_in)),
                       # showmeans=True,
                       showmedians=True
                       )

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(list(colors.values())[i])
    # pc.set_edgecolor('black')
    # pc.set_linewidth(2)
    
    pc.set_alpha(1)
    pc.set_edgecolor('#262626')

for i,j in parts.items():
    if i == 'bodies':
        continue
    else:
        parts[i].set_colors('#262626')
        parts[i].set_linewidth(3)


"""
lifetimes
"""

diffusion = pd.read_csv(f'{base}/processed_data/lifetimes/residence_exponents.csv').sort_values('exponent', ascending=False)
cols = [colors[i] for i in diffusion['class']]

ax1.bar(
       cl_labels, #['\n'.join(i.split('_')) for i in diffusion['class']],
       diffusion['exponent'],
       color=cols,
       yerr=diffusion['exponent_err'],
       )


#####################
### LOAD THE DATA ###
#####################

datasets = [pickle.load(open(f"{base}/analysed_data/with_metabolites/replica_{i}/binding_{i}.pkl", 'rb')) for i in range(1,4)]

proteomics = pd.read_csv(f'{base}/processed_data/proteomics_annotated.csv',
                         index_col='Locus tag')

# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(io.StringIO(s.decode('utf-8')),
                          usecols=['Metabolite name', 
                                   'resname',
                                   'class'],
                          index_col='Metabolite name')

metabolite_classes = metabolites.to_dict()['class']

# for each replica, for each protein, get the number of normalised counts for ATP.
# we end up with a dictionary of {protein: [normalised_count_0, normalised_count_1... normalised_count_n]}
ATP_results = defaultdict(list)
for idx, d in enumerate(datasets):
    ATP_vals = {protein: value.get('ATPH', {}).get('normalised_count', 0) for protein, value in d['protein_results'].items()}
    for protein, val in ATP_vals.items():
        ATP_results[protein].append(val)

# from the proteomics, make a dictionary indicating whether the each protein is a known binder of ATP or not
ATP_mask = {}
for protein in proteomics.T:
    ligs = proteomics.loc[protein]['Ligands']
    protein_renamed = protein[-3:]+'_monomer'
    if 'ATP' in str(ligs):
        ATP_mask[protein_renamed] = True
    else:
        ATP_mask[protein_renamed] = False

# combine the results with the mask
arr = []
names = []
mask = []
for key, val in ATP_results.items():
    if key in ATP_mask.keys():
        mask.append(ATP_mask[key])
        arr.append(np.array([np.mean(val), np.std(val)]))
        names.append(key)
arr = np.array(arr)

# sort the results so that we can plot them in order
sorter = np.argsort(arr.T[0])
data = arr.T[0][sorter]
err =  arr.T[1][sorter]
names_sorted = np.array(names)[sorter]
x_plt = np.arange(len(names_sorted))
mask_sorted = np.array(mask)[sorter]

# make the plot
# fig, ax = plt.subplots(figsize = (15,10))

# plot the binders
ax2.bar(x_plt[mask_sorted],
       data[mask_sorted],
       yerr=err[mask_sorted],
       error_kw = {'elinewidth':.75,
                   'ecolor': '#944CE5'},
       color='#944CE5',
       label='Known binders',
       width=1,
       align='center'
       )
# plot the non binders
ax2.bar(x_plt[~mask_sorted],
       data[~mask_sorted],
       yerr=err[~mask_sorted],
       error_kw = {'elinewidth':.75,
                   'ecolor': '#3C3B30',
                   'alpha':0.5
                   },
       color='#3C3B30',
       label='Non binders',
       align='center',
       alpha=0.5,
       )


# set things up for statistical testing
binding_values = arr.T[0]
binding_errors = arr.T[1]
labels = mask

# mask values
known_binders = binding_values[np.where(np.array(labels) == True)[0]]
non_binders = binding_values[np.where(np.array(labels) == False)[0]]
# errs
known_binders_err = binding_errors[np.where(np.array(labels) == True)[0]]
non_binders_err = binding_errors[np.where(np.array(labels) == False)[0]]

mw_iters = 50000

U_vals, p_vals = bootstrap_mannwhitney(
    known_binders, known_binders_err,
    non_binders, non_binders_err,
    n_iter=mw_iters,
    alternative="greater",
)
median_p = np.median(p_vals)
frac_significant = np.mean(p_vals < 0.05)


mannwhitney_report = []

mannwhitney_report.append(f"No. known ATP binders: {len(known_binders)}")
mannwhitney_report.append(f"Known ATP binders av. associations: {known_binders.mean():.2f}\n")

mannwhitney_report.append(f"No. ATP non-binders: {len(non_binders)}")
mannwhitney_report.append(f"ATP non-binders av. associations: {non_binders.mean():.2f}\n")

mannwhitney_report.append(f"No. bootstrap sample tests: {mw_iters}")
mannwhitney_report.append(f"Median p-value: {median_p:.3g}")
mannwhitney_report.append(f"Fraction p < 0.05: {frac_significant:.2f}")

p_lo, p_hi = np.percentile(p_vals, [16, 84])
mannwhitney_report.append(f"68% interval on p-value: [{p_lo:.3g}, {p_hi:.3g}]")

U_nominal, p_nominal = mannwhitneyu(known_binders, non_binders, alternative="greater")
mannwhitney_report.append(f"Nominal p-value (ignoring errors): {p_nominal:.3e}")

with open("mannwhitney_report.txt", 'w') as f:
    f.writelines('\n'.join(mannwhitney_report))


# make the inset plot of the p-values
axins = inset_axes(ax2, width="75%", height="75%",
                   bbox_to_anchor=(.075, .55, .5, .5),
                   bbox_transform=ax2.transAxes, 
                   loc="lower left")

bins=np.array([0,0.025,0.05,0.1,0.2,0.3,0.6])

axins.hist(p_vals, bins=bins,
           edgecolor='#262626',
           linewidth=2,
           histtype='step',
           log=True)
axins.axvline(0.05, 
              c='#262626',
              linestyle="--")
axins.set_xlim(0,axins.get_xlim()[1])
axins.set_xlabel("p-value")
axins.set_ylabel("Count")
for spine in axins.spines.values():
    spine.set_edgecolor('#262626')
    spine.set_linewidth(1)
    
ax2.legend(loc='upper left',
          bbox_transform=axins.transAxes,
          bbox_to_anchor=(1,1),
          fontsize=30)


"""
plot aesthetics
"""
fig.subplots_adjust(hspace=0.1)


ax0.set_yscale('log')
ax0.set_ylabel('Diffusion (cm$^2$/s)', fontsize=40)


ax0.tick_params(labelsize=20)
# ax0.set_ylim(2e0,10e2)
ax0.set_xticks(np.arange(len(diffusion)),
                 ['']*len(diffusion))
    
for i,j in enumerate(cl_labels): #['\n'.join(i.split('_')) for i in diffusion['class']]):
    ax1.text(i,0.1,
             j,
             fontsize=22.5,
             ma='center',
             va='center',
             ha='center'
             )
ax1.set_xticks(np.arange(len(diffusion)),
               ['']*len(diffusion))

ax1.tick_params(axis='y',
                labelsize=20,
                labeltop=True,
                labelbottom=False)

ax1.set_ylabel('Association lifetime\ndistribution, power\nlaw exponent', fontsize = 40)


ax2.set_xticks([])
ax2.set_xlim(-0.75,np.arange(len(names_sorted))[-1]+0.75)
ax2.set_ylabel('Asssociation events\nper protein per ATP', 
              fontsize = 40)
ax2.tick_params(labelsize=20)
ax2.set_ylim(0,4.2)


fig.savefig("figure_3.png",
            dpi=500,
            bbox_inches='tight')





