#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:07:22 2026

@author: chrisbrasnett
"""

import MDAnalysis as mda
from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
from glob import glob
from tqdm import tqdm
from pathlib import Path
import pickle

fs = glob('proteins/*.pdb')

results = {}
for protein in tqdm(fs):
    u = mda.Universe(protein)
    analysis = SASAAnalysis(u)
    analysis.run()
    total_area = analysis.results['total_area']
    results[Path(protein).stem] = total_area[0]

pickle.dump(dict(sorted(results.items())), open('protein_sasa.pkl', 'wb'))
