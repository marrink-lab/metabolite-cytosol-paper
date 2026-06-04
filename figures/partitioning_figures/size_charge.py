#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:35:36 2026

@author: chrisbrasnett
"""

import numpy as np
import matplotlib.pyplot as plt

from lmfit.models import LinearModel
from scipy.linalg import LinAlgError
import requests
import io
import pandas as pd
from uncertainties import ufloat_fromstr
from uncertainties import unumpy
from rdkit import Chem

plt.style.use("../../mystyle.mplstyle")


def plotter(_x,_y,plotter_lim):
    
    x_sorted = np.sort(_x)
    y_sorted = _y[np.argsort(_x)]
    
    x = x_sorted[~np.isnan(x_sorted)]
    y = y_sorted[~np.isnan(x_sorted)]
    
    mod = LinearModel()
    try:
        pars = mod.guess(y, x=x)
    except LinAlgError:
        return None
    
    res = mod.fit(y, params=pars, x=x)
    
    plotter = np.linspace(plotter_lim[0], plotter_lim[1], 1000)
    res_x = res.eval(x=plotter)
    res_uncert = res.eval_uncertainty(x=plotter)
    
    return np.stack((plotter, res_x, res_uncert))
    

# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(io.StringIO(s.decode('utf-8')),
                          usecols=[
                              'resname',
                              'class',
                              'SASA CG',
                              'Canonical SMILES'
                              ],
                          index_col='resname')

partitioning_data = 'partitioning_per_class_per_mol.xlsx'

counts = 'table_S1_cytosol_composition.xlsx'

all_sheets = pd.read_excel(partitioning_data, sheet_name=None, index_col=0)

df = pd.concat(all_sheets.values(), ignore_index=False)
counts_df = pd.read_excel(counts, sheet_name='Metabolites', index_col=0, )

merged_df = df.join(metabolites, how="left").join(counts_df, how='left')
charges = []
for i in merged_df['Canonical SMILES']:
    try:
        mol = Chem.MolFromSmiles(i)
        total_charge = Chem.GetFormalCharge(mol)
    except TypeError:
        total_charge = np.nan
    charges.append(total_charge)

merged_df['charge'] = charges

merged_df['Count mask'] = merged_df['Count'] > 1
fig, axarr = plt.subplots(3,2,sharex='col',
                          sharey=True,
                       figsize=(10,10)
                       )
COLORS = {'protein_adsorbed': '#8C64C8', 
          'soluble': '#D26496', 
          'clustered': '#FF9664'}
LABELS = {'protein_adsorbed': 'Protein adsorbed', 
          'soluble': 'Soluble', 
          'clustered': 'Clustered'}

CLASS_COLS = {
    'Amino_Acids':       "#F9B714",  # bright warm yellow
    'Ions':              "#337AEA",  # bright cobalt
    'Lipids':            "#05938E",  # jade teal
    'Cofactors':         "#B7C61E",  # bright olive
    'Other_Metabolites': "#CC7FDB",  # bright mauve
    'Nucleotides':       "#944CE5",  # bright violet
    'Carbohydrates':     "#3FB760",  # bright fern
}

CLASS_LABELS = {
    'Amino_Acids': 'Amino acids', 
    'Carbohydrates': 'Carbohyd.', 
    'Cofactors': 'Cofactors',
    'Ions': 'Ions', 
    'Lipids': 'Fatty acid', 
    'Nucleotides': 'Nucleotides', 
    'Other_Metabolites': 'Other',
}

for i,j in enumerate(['protein_adsorbed', 'soluble', 'clustered']):
    for cat, group in merged_df.groupby("class"):
        vals = np.array([ufloat_fromstr(k) for k in group[j].values])
        
        mask = group['Count mask']
        
        axarr[i][0].errorbar(group['charge'][mask],
                             unumpy.nominal_values(vals)[mask],
                             yerr=unumpy.std_devs(vals)[mask],
                             ls='none',
                             marker='.',
                             markersize=10,
                             markeredgewidth=.5,
                             markeredgecolor='#262626',
                             c=CLASS_COLS.get(cat)
                             )
        axarr[i][1].errorbar(group['SASA CG'][mask],
                             unumpy.nominal_values(vals)[mask],
                             yerr=unumpy.std_devs(vals)[mask],
                             ls='none',
                             marker='.',
                             markersize=10,
                             markeredgewidth=.5,
                             markeredgecolor='#262626',
                             c=CLASS_COLS.get(cat),
                             label=CLASS_LABELS.get(cat)
                             )

    
    axarr[i][0].set_ylabel(LABELS[j],
                           c = COLORS[j],
                           fontweight='bold')
    axarr[i][0].set_ylim(-0.04,1)
    axarr[i][1].set_ylim(-0.04,1)

    axarr[i][0].set_xlim(-5.2,4.5)
    axarr[i][1].set_xlim(1.5,12)

    
    x0 = merged_df['charge'].values[merged_df['Count mask']]
    x1 = merged_df['SASA CG'].values[merged_df['Count mask']]
    y = np.array([ufloat_fromstr(k).n for k in merged_df[j].values])[merged_df['Count mask']]
    
    r0 = plotter(x0, y, [-6,5])
    r1 = plotter(x1, y, [0,12])
    if r0 is not None:
        axarr[i][0].plot(r0[0],r0[1],c='#262626',ls='--')
        axarr[i][0].fill_between(r0[0],
                                 r0[1] - r0[2],
                                 r0[1] + r0[2],
                                 color='#262626',
                                 alpha=.3
                                 )
    if r1 is not None:
        axarr[i][1].plot(r1[0],r1[1],c='#262626',ls='--')
        axarr[i][1].fill_between(r1[0],
                                 r1[1] - r1[2],
                                 r1[1] + r1[2],
                                 color='#262626',
                                 alpha=.3
                                 )
    

axarr[-1][0].set_xlabel('Charge')
axarr[-1][1].set_xlabel(r'SASA CG (nm$^2$)')

axarr[-1][-1].legend(ncol=1,loc='center left',
                     bbox_transform=fig.transFigure,
                     # bbox_to_anchor=(0.5,0.05),
                     bbox_to_anchor=(.9,0.5),
                     
                     fontsize=20
                     )

fig.subplots_adjust(hspace=.2,wspace=.1)

fig.savefig('partition_correlation.png',
            dpi = 300,
            bbox_inches='tight')





