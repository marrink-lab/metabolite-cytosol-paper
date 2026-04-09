#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:27:56 2026

@author: chrisbrasnett
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import pandas as pd
from lmfit.models import ExponentialModel, PowerLawModel
from collections import OrderedDict
plt.style.use("/Users/chrisbrasnett/mystyle.mplstyle")

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


fig, axarr = plt.subplots(nrows=3, ncols=2,
                          sharex='row',
                          sharey=True,
                          figsize=(15,15))
rsq=r"R$^2$" # for formatting later

# iterate over each row of the plot with some associated useful stuff
for ax_row, indep_var, lims, ylab in zip(axarr, 
                                   [MWs, sasas, spheres], 
                                   [1800,20,1.5],
                                   ['Molecular weight (g/mol)', 
                                    r'SASA (nm$^2$)', 
                                    r'Radius ($\sqrt{\frac{\mathrm{SASA}}{4\pi}}$) (nm)']
                                   ):
    # iterate over each row
    for col_idx, ax in enumerate(ax_row):
        
        # set up some lists to put the data in
        x_fit = []
        y_fit = []
        
        lipids_x = []
        lipids_y = []
        
        # iterate over the classes we have
        for cl, col in colors.items():
            # we want to take care over what we do with the lipids
            if cl == "Lipids":
                # don't plot the lipids in the first column of the figure
                if col_idx == 0:
                    continue
                else:
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
                                    label=' '.join(cl.split('_'))
                                    )
            # plot everything in the second column
            else:
                for mol, D_arr in diffusion_data[cl].items():
                    xs = indep_var.get(mol)
                    ys = D_arr[0]
                    yerrs = D_arr[1]
                    
                    x_fit.append(xs)
                    y_fit.append(ys)
                    
                    ax.errorbar(xs, 
                                ys, yerr=yerrs,
                                ls='',
                                marker = '.', markersize=25,
                                alpha=0.7,
                                markeredgecolor='#262626', markeredgewidth=1,
                                c=col, 
                                label=' '.join(cl.split('_'))
                                )
        
        # fit the data that we've collected
        plotter_lim = lims
        _x = np.array(x_fit)[np.argsort(np.array(x_fit))]
        _y = np.array(y_fit)[np.argsort(np.array(x_fit))]
        fit_result = fitting(_x, _y, plotter_lim)
        
        # if we've collected data for the lipids only, also make a fit to that
        if len(lipids_x) > 0:
            _lipids_x = np.array(lipids_x)[np.argsort(np.array(lipids_x))]
            _lipids_y = np.array(lipids_y)[np.argsort(np.array(lipids_x))]
            
            mod = PowerLawModel()
            pars = mod.guess(_lipids_y, x=_lipids_x)
            result = mod.fit(_lipids_y, params=pars, x=_lipids_x)
            modelname = result.model.name.split(')')[0].split('(')[1]
            r_squared = result.rsquared
    
            plotter = np.linspace(0.1, plotter_lim, 1000)
            res_x = result.eval(x=plotter)
            res_uncert = result.eval_uncertainty(x=plotter)
    
            fit_result["LipidsPowerLaw"] = {'xplt': plotter,
                                            'res': res_x,
                                            'err': res_uncert,
                                            'r2': r_squared,
                                            'col': '#262626',
                                            'label': 'exponent',
                                            'label_val': result.params['exponent']
                                            }
        
        # plot the fits that we've made
        for modelname, fit_data in fit_result.items():
            ax.plot(fit_data['xplt'],
                    fit_data['res'],
                    ls='--',
                    c=fit_data['col'],
                    label=f"{modelname}\nexponent = {fit_data['label_val'].value:.2f} ± {fit_data['label_val'].stderr:.2f}\n{rsq} = {fit_data['r2']:.3f}"
                    )
            ax.fill_between(fit_data['xplt'],
                            fit_data['res'] - fit_data['err'],
                            fit_data['res'] + fit_data['err'],
                            color=fit_data['col'],
                            alpha=0.5)
        
        # sort out the figure legend so we only have one entry for each class etc.
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        
        # to make the fits and their data appear in the legend
        if col_idx == 0:
            leg = ax.legend(list(by_label.values())[:2], 
                            list(by_label.keys())[:2],
                            )
        else:
            leg = ax.legend(list(by_label.values())[:3], 
                            list(by_label.keys())[:3],
                            )
        ax.add_artist(leg)

        # plot aesthetics
        ax.set_ylim(10e-2,10e3)
        ax.set_yscale('log')
        ax.set_xlabel(ylab)
        
# add the metabolite classes
leg = axarr[-1,-1].legend(list(by_label.values())[3:], 
                          list(by_label.keys())[3:],
                          ncol=3, loc = 'upper center',
                          bbox_to_anchor=(0.5,0.05),
                          bbox_transform=fig.transFigure,
                          fontsize=20
                          )
# &c.
ylabs = [i[0].set_ylabel('Diffusion (x10$^{-9}$ m$^2$/s)', fontsize=20) for i in axarr]
fig.subplots_adjust(wspace=0.1, hspace = 0.3)


fig.savefig('diffusion_fits.png',
            dpi = 500,
            bbox_inches='tight')



