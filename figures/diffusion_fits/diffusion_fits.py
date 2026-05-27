#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import pandas as pd
from lmfit.models import ExponentialModel, PowerLawModel
from collections import OrderedDict
plt.style.use("../../mystyle.mplstyle")

"""
load some data
"""

base = '../..'

# get metabolite classes from the master database spreadsheet on the M3-metabolome repo
metabolites_url = "https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Metabolome/refs/heads/main/misc/database.csv"
s = requests.get(metabolites_url).content

metabolites = pd.read_csv(io.StringIO(s.decode('utf-8')),
                          usecols=[
                              'resname',
                              'class',
                              'MW',
                              'SASA CG'
                              ],
                          index_col='resname')

# make some dicts of molecular properties from the database
MWs = metabolites.to_dict()['MW']
sasas = metabolites.to_dict()['SASA CG']
spheres = {i:np.sqrt(j/(4*np.pi)) for i,j in sasas.items()}

# the diffusion data generated from MartiniSoup
diffusion_data = pickle.load(open(f'{base}/processed_data/diffusion/diffusion_coefficients.pkl', 'rb'))['results']
# set the colours
colors = {
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


def fitting(_x, _y, plotter_lim):
    """
    
    function to fit a power law and exponential model to input data.
    
    return a dict with plotting information that we need
    
    """
    x = _x[~np.isnan(_x)]
    y = _y[~np.isnan(_x)]
    
    plotter = np.linspace(0.1, plotter_lim, 1000)
    
    d_out = {}
    for mod, col, param in zip([PowerLawModel(), ExponentialModel()],
                    ['#E6B60E', '#0E22E6'],
                    ['exponent', 'decay']):
        pars = mod.guess(y, x=x)
        result = mod.fit(y, params=pars, x=x)
        modelname = result.model.name.split(')')[0].split('(')[1]
        r_squared = result.rsquared
        print(result.fit_report())
        
        res_x = result.eval(x=plotter)
        res_uncert = result.eval_uncertainty(x=plotter)
        
        d_out[modelname] = {'xplt': plotter,
                            'res': res_x,
                            'err': res_uncert,
                            'r2': r_squared,
                            'col': col,
                            'label': param,
                            'label_val': result.params[param]
                            }
    return d_out


# fig, axarr = plt.subplots(ncols=3,
#                           # sharex='row',
#                           sharey=True,
#                           figsize=(15,7.5)
#                           )
rsq=r"R$^2$" # for formatting later

# iterate over each row of the plot with some associated useful stuff
# for ax, indep_var, lims, ylab in zip(axarr, 
#                                    [MWs, sasas, spheres], 
#                                    [900,13,1.2],
#                                    ['Molecular weight (g/mol)', 
#                                     r'SASA (nm$^2$)', 
#                                     r'Radius ($\sqrt{\frac{\mathrm{SASA}}{4\pi}}$) (nm)']
#                                    ):
# set up some lists to put the data in

fig, ax = plt.subplots()
indep_var = spheres
lims = 1.2
ylab=r'Radius ($\sqrt{\frac{\mathrm{SASA}}{4\pi}}$) (nm)'

x_fit = []
y_fit = []

lipids_x = []
lipids_y = []

# iterate over the classes we have
for cl, col in colors.items():
    for mol, D_arr in diffusion_data[cl].items():
        xs = indep_var.get(mol)
        ys = D_arr[0]
        yerrs = D_arr[1]
        
        x_fit.append(xs)
        y_fit.append(ys)
        
        # collect a lipid specific dataset here
        lipids_x.append(xs)
        lipids_y.append(ys)
        
        ax.errorbar(xs, 
                    ys, yerr=yerrs,
                    ls='',
                    marker = '.', markersize=25,
                    alpha=0.7,
                    markeredgecolor='#262626', markeredgewidth=1,
                    c=col, 
                    label=CLASS_LABELS[cl]
                    )

# fit the data that we've collected
plotter_lim = lims
_x = np.array(x_fit)[np.argsort(np.array(x_fit))]
_y = np.array(y_fit)[np.argsort(np.array(x_fit))]
fit_result = fitting(_x, _y, plotter_lim)

model_names = {'powerlaw': 'Power Law',
               'exponential': 'Exponential Decay'}

# plot the fits that we've made
for modelname, fit_data in fit_result.items():
    mod_name=model_names[modelname]
    ax.plot(fit_data['xplt'],
            fit_data['res'],
            ls='--',
            c=fit_data['col'],
            label=f"{mod_name}\nexponent = {fit_data['label_val'].value:.2f} ± {fit_data['label_val'].stderr:.2f}\n{rsq} = {fit_data['r2']:.3f}"
            )
    ax.fill_between(fit_data['xplt'],
                    fit_data['res'] - fit_data['err'],
                    fit_data['res'] + fit_data['err'],
                    color=fit_data['col'],
                    alpha=0.5)

# sort out the figure legend so we only have one entry for each class etc.
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))

leg = ax.legend(list(by_label.values())[:2], 
                list(by_label.keys())[:2],
                )
ax.add_artist(leg)

# plot aesthetics
ax.set_xlim(0.3, 1)
ax.set_ylim(10e-7,5e-3)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(ylab)


# add the metabolite classes
# leg = axarr[0].legend(list(by_label.values())[2:], 
#                           list(by_label.keys())[2:],
#                           ncol=4, loc = 'upper center',
#                           bbox_to_anchor=(0.5,0),
#                           bbox_transform=fig.transFigure,
#                           fontsize=20
#                           )

leg = ax.legend(list(by_label.values())[2:], 
                          list(by_label.keys())[2:],
                          ncol=1, loc = 'center left',
                          bbox_to_anchor=(.9,0.5),
                          bbox_transform=fig.transFigure,
                          fontsize=20
                          )

ylabs = ax.set_ylabel('Diffusion (cm$^2$/s)', fontsize=20)
# fig.subplots_adjust(wspace=0.1, hspace = 0.3)

fig.savefig('diffusion_fit_radius.png',
            dpi = 500,
            bbox_inches='tight')



