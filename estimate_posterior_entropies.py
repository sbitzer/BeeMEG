#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:22:13 2016

@author: bitzer
"""
import os
import numpy as np

fname = 'infres_collapse_201612121759.npz'
#fname = 'infres_basic_201612121421.npz'
resultsdir = os.path.expanduser('~/ZIH/gitrepos/BeeMEG/data/inf_results')

S = 15000
entropy_bins = 50

with np.load(os.path.join(resultsdir, fname)) as data:
    subjects = data['subjects']
    model = data['model'][()]
    pars = data['pars'][()]
    ep_means = data['ep_means']
    ep_covs = data['ep_covs']
    
entropies = np.zeros((subjects.size, model.L))
for si, sub in enumerate(subjects):
    print('subject %d ...' % sub, flush=True)
    for tri, tr in enumerate(np.arange(model.L)):
        # sample from posterior
        choices, rts = model.gen_response_from_Gauss_posterior(tr, 
            pars.names, ep_means[si, :], ep_covs[si, :, :], S, pars.transform)
        
        # estimate entropy
        entropies[si, tri] = model.estimate_entropy(choices, rts, entropy_bins)
        
    np.savez_compressed(os.path.join(resultsdir, fname[:-4]+'_entropies.npz'), 
                        entropies=entropies, S=S, entropy_bins=entropy_bins)