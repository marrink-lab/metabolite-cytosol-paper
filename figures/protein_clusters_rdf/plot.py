#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:43:40 2026

@author: chrisbrasnett
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import networkx as nx

from scipy.ndimage import gaussian_filter1d

plt.style.use("/Users/chrisbrasnett/mystyle.mplstyle")

def replicate_analysis(replicate):
    # Create networkx graph from the contact data
    G = nx.Graph()

    # Add nodes
    for node in replicate['nodes']:
        G.add_node(node['id'], type=node['type'])

    # Add edges (contacts)
    for link in replicate['links']:
        G.add_edge(link['source'], link['target'])

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Calculate cluster statistics
    n_clusters = len(clusters)
    cluster_sizes = [len(cluster) for cluster in clusters]
    largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

    return {
        'time': replicate['time'] / 1000,
        'frame': replicate['frame'],
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'largest_cluster_size': largest_cluster_size,
        'avg_cluster_size': avg_cluster_size,
        'total_proteins': len(replicate['nodes']),
        'total_contacts': len(replicate['links']) // 2  # Divide by 2 for undirected
    }

def analyze_clusters(contacts_data):
    """Analyze clusters for each frame"""
    cluster_stats = [replicate_analysis(replicate) for replicate in contacts_data]
            
    frames = [i.get('time') for i in cluster_stats][:250]
    avg_cluster_size = [i.get('avg_cluster_size') for i in cluster_stats][:250]

    return np.stack((frames, avg_cluster_size))

def cluster_plotter(path, label, ax, col):
    
    results_files = sorted(glob.glob(path))
    results_list  = [pickle.load(open(f, 'rb')) for f in results_files]
    
    contacts_data = [analyze_clusters(i['frames']) for i in results_list]
    
    x = gaussian_filter1d(np.stack(contacts_data).mean(axis = 0)[0],
                          3)
    y = gaussian_filter1d(np.stack(contacts_data).mean(axis = 0)[1],
                          3)
    yerr = gaussian_filter1d(np.stack(contacts_data).std(axis = 0)[1],
                             3)
    ax.plot(x,
            y,
            label = label,
            color=col
            )
    
    ax.fill_between(x,
                    y - yerr,
                    y + yerr,
                    alpha=0.2,
                    color=col,
                    )

def rdf_plot(path, ax):
    
    labs = {'with_metabolites': 'With metabolites',
            'without_metabolites': 'Without metabolites',
            }
    cols = {'with_metabolites': '#05938E',
            'without_metabolites': '#666666',
            }
    data = pickle.load(open(path, 'rb'))
    
    for i,j in data.items():
        ax.plot(j['r']/10,
                gaussian_filter1d(j['gr_mean'], 3),
                label=labs[i],
                c = cols[i]
                )
        ax.fill_between(j['r']/10,
                gaussian_filter1d(j['gr_mean'], 3) - gaussian_filter1d(j['gr_std'], 3),
                gaussian_filter1d(j['gr_mean'], 3) + gaussian_filter1d(j['gr_std'], 3),
                # label=labs[i],
                color = cols[i],
                alpha=.2
                )
        
    ax.legend()

    

fig, (ax1,ax2) = plt.subplots(    1, 2, figsize=(10, 4),
    gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.25}
)
ax1.set_xlabel('Time (ns)', fontsize=15)
ax1.set_ylabel('Mean cluster size (no. of proteins)', fontsize=15)

cluster_plotter("../../analysed_data/with_metabolites/*/protein_clusters_*.pkl",
        'With metabolites',
        ax1, col='#05938E')
cluster_plotter("../../analysed_data/no_metabolites/*/protein_clusters_*.pkl",
        'Without metabolites',
        ax1, col='#666666')
ax1.legend()
ax1.set_xlim(left=0)
ax1.legend(loc='upper left', frameon=False, fontsize=10.5)


rdf_plot('protein_rdf.pkl',
         ax2
         )
         # ax2,
         # color='#05938E',label='With metabolites')

# rdf_plot("../../analysed_data/no_metabolites/*/protein_rdf_*.pkl",
#          ax2,
#          color='#666666',label='Without metabolites')

ax2.axhline(1.0, color='#BBBBBB', lw=0.8, ls='--', zorder=1)
ax2.set_xlabel('r (nm)', fontsize=15)
ax2.set_ylabel('g(r)', fontsize=15)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1.75)
ax2.legend(loc='upper left', frameon=False, fontsize=10.5)


# ── Shared style ──────────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.savefig('clustering_and_rdf.png', dpi=300, bbox_inches='tight', facecolor='white')
