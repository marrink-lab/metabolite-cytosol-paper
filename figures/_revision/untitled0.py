#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lmfit.models import ExponentialModel, PowerLawModel
from rdkit import Chem
from uncertainties import ufloat, ufloat_fromstr

plt.style.use("../../mystyle.mplstyle")

"""
load some data
"""

base = "../.."

# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(
    io.StringIO(s.decode("utf-8")),
    usecols=["resname", "Canonical SMILES", "Measured logP", "class"],
    index_col="resname",
)

# the diffusion data generated from MartiniSoup
diffusion_data = pickle.load(
    open(f"{base}/processed_data/diffusion/diffusion_coefficients.pkl", "rb")
)["results"]
# set the colours
COLORS = {
    "Amino_Acids": "#F9B714",  # bright warm yellow
    "Ions": "#337AEA",  # bright cobalt
    "Lipids": "#05938E",  # jade teal
    "Cofactors": "#B7C61E",  # bright olive
    "Other_Metabolites": "#CC7FDB",  # bright mauve
    "Nucleotides": "#944CE5",  # bright violet
    "Carbohydrates": "#3FB760",  # bright fern
}

CLASS_LABELS = {
    "Amino_Acids": "Amino acids",
    "Carbohydrates": "Carbohyd.",
    "Cofactors": "Cofactors",
    "Ions": "Ions",
    "Lipids": "Fatty acid",
    "Nucleotides": "Nucleotides",
    "Other_Metabolites": "Other",
}


_logp = metabolites.to_dict()['Measured logP']
_charges = metabolites.to_dict()['Canonical SMILES']
classes = metabolites.to_dict()['class']
diffusion_keys = [key0 for key in diffusion_data.keys() for key0 in dict(diffusion_data[key]).keys()]


logp = {}
for key, value in _logp.items():
    if value is not np.nan:
        logp[key] = ufloat_fromstr(value)

charges = {}
for key, value in _charges.items():
    try:
        mol = Chem.MolFromSmiles(value)
        net_charge = Chem.GetFormalCharge(mol)
        
        charges[key] = net_charge
    except TypeError:
        # print(key, value)
        pass

fig, (ax0, ax1) = plt.subplots(1,2, sharey=True,
                               figsize=(15,7.5))

# data = []
for key, value in logp.items():
    if key in diffusion_keys:
        x_val = value.n
        x_err = value.s
        
        
        cl = classes[key]
        
        diffusion = diffusion_data[cl][key]
        
        y_val = diffusion[0]
        y_err = diffusion[1]
        
        # data.append([x_val, y_val])
        
        ax0.errorbar(x_val,
                    y_val,
                    xerr = x_err,
                    yerr = y_err,
                    color = COLORS[cl],
                    ls="",
                    marker=".",
                    markersize=25,
                    alpha=0.7,
                    markeredgecolor="#262626",
                    markeredgewidth=1,
                    label=cl
                    )

ax0.set_yscale('log')
ax0.set_xlabel(r'$\Delta$G$_{OW}$ (kJ/mol)')
ax0.set_ylabel("Diffusion (cm$^2$/s)", fontsize=20)

for key, value in charges.items():
    if key in diffusion_keys:
        x_val = value
        
        cl = classes[key]
        
        diffusion = diffusion_data[cl][key]
        
        y_val = diffusion[0]
        y_err = diffusion[1]
        
        ax1.errorbar(x_val,
                    y_val,
                    # xerr = x_err,
                    yerr = y_err,
                    color = COLORS[cl],
                    ls="",
                    marker=".",
                    markersize=25,
                    alpha=0.7,
                    markeredgecolor="#262626",
                    markeredgewidth=1,
                    label=cl
                    )

ax1.set_yscale('log')
ax1.set_xlabel("Charge")

handles, labels = ax1.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))

ax1.legend(
    list(by_label.values()),
    [' '.join(i.split('_')) for i in list(by_label.keys())],
    loc='center left',
    bbox_transform = ax1.transAxes,
    bbox_to_anchor=(1,0.5)
)


fig.subplots_adjust(wspace=.1)

fig.savefig("diffusion_logp_charge.png",
            dpi=200,
            bbox_inches="tight")
