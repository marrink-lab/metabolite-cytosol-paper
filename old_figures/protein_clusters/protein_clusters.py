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
from tqdm import tqdm
from scipy.signal import savgol_filter

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
            
    frames = [i.get('time') for i in cluster_stats]
    avg_cluster_size = [i.get('avg_cluster_size') for i in cluster_stats]

    return np.stack((frames, avg_cluster_size))

def plotter(path, label, ax):
    
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
            label = label
            )
    
    ax.fill_between(x,
                    y - yerr,
                    y + yerr,
                    alpha=0.5
                    )
    
fig, ax = plt.subplots()
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Mean cluster size (no. of proteins)')

plotter("../../analysed_data/*/protein_contacts_*.pkl",
        'with metabolites',
        ax)
plotter("../../analysed_data/no_metabolites/*/protein_contacts_*.pkl",
        'no metabolites',
        ax)
ax.legend()

