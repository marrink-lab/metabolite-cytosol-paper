#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:05:27 2026

@author: chrisbrasnett
"""

import numpy as np
from vermouth.gmx.itp_read import read_itp
from vermouth.forcefield import ForceField
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("../../mystyle.mplstyle")

def get_net_charges(path, filter=False):

    ff = ForceField('dummy')
    
    with open(path) as f:
        lines = f.readlines()
        
    read_itp(lines, ff)
    
    net_charges = {}
    for block in ff.blocks:
        net_charge = sum([ff.blocks[block].nodes[i].get('charge',0) for i in ff.blocks[block].nodes])
        net_charges[block] = net_charge
    
    if filter:
        counts = '../../misc/table_S1_cytosol_composition.xlsx'
        valid_proteins = pd.read_excel(counts, sheet_name='Proteins')['Molecule name'].values
        net_charges = {key: value for key, value in net_charges.items() if key in valid_proteins}
    
    net_charges_arr = np.array(list(net_charges.values()))
    
    return net_charges_arr

def make_plot(ax, net_charges_arr,bins):
    ax.hist(net_charges_arr, bins = bins)
    ax.axvline(0,c='#262626', ls='--')
    # ax.set_xlabel('Net charge')
    ax.set_ylabel('Count')
    ax.text(1,1,
            f'{sum(net_charges_arr>0)} > 0',
            transform = ax.transAxes,
            ha='right',
            va='top',
            fontsize=20)
    ax.text(0,1,
            f'{sum(net_charges_arr<0)} < 0',
            transform = ax.transAxes,
            va='top',
            fontsize=20)
    print(sum(net_charges_arr==0))

proteins = get_net_charges('/Users/chrisbrasnett/Downloads/proteins.itp', filter=True)
metabolites = get_net_charges('/Users/chrisbrasnett/Downloads/metabolites.itp')

fig, (ax0,ax1) = plt.subplots(2,1,figsize=(10,10))
make_plot(ax0, proteins,50)
make_plot(ax1, metabolites,10)

ax0.set_xlabel('Protein net charge')
ax1.set_xlabel('Metabolite net charge')


fig.savefig('net_charges.png', dpi = 300, bbox_inches='tight')


