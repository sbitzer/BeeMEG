#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:11:08 2017

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import subject_DM
import pymc3 as pm
from scipy.stats import norm
import os


#%% load all data
epochs_all = helpers.load_meg_epochs(hfreq=10, window=[0.4, 0.7])
subjects = epochs_all.index.levels[0]
S = subjects.size

times = epochs_all.index.levels[2]


#%% load trial-level design matrices for all subjects (5th-dot based)
DM = subject_DM.get_ithdot_DM()
DM = DM.loc(axis=0)[subjects, :]
R = DM.shape[1]

# remove trials which have nan
nonan_idx = np.logical_not(np.isnan(DM['surprise'])).values
DM = DM.loc[nonan_idx]

sub_map = {x: np.flatnonzero(subjects==x)[0] for x in subjects}
subject_idx = DM.index.get_level_values('subject').map(lambda x: sub_map[x])


#%% 
# 2nd-level posterior probabilities
def post_prob(mean, sd):
    if mean > 0:
        return norm.cdf(0, mean, sd)
    else:
        return 1 - norm.cdf(0, mean, sd)


#%% define hierarchical GLM in pyMC3
time = 500
channel = 'MEG2041'
pvals = pd.DataFrame([], index=pd.MultiIndex.from_product([epochs_all.columns, 
                     epochs_all.index.levels[2]], names=['channel', 'time']), 
                     columns=DM.columns, dtype=np.float64)

for time in times:
    for channel in epochs_all.columns:
        data = epochs_all.loc[(slice(None), slice(None), time), channel]
        data = data.loc[nonan_idx]

        with pm.Model() as model:
            beta2 = pm.Normal('beta2', sd=100**2, shape=R)
            sigma2 = pm.HalfCauchy('sigma2', 5, shape=R, testval=.1)
            
            beta1 = pm.Normal('beta1', mu=beta2, sd=sigma2, shape=(S, R))
            
            eps = pm.HalfCauchy('eps', 5)
            
            pred = (beta1[subject_idx, :] * DM.values).sum(axis=1)
            
            likelihood = pm.Normal(channel, mu=pred, sd=eps, observed=data.values)
            
            means, sds, elbos = pm.variational.advi(n=40000)
        
        # compute "p-values"
        pvals.loc[(channel, time), :] = np.array([post_prob(m, s) for m, s in 
                 zip(means['beta2'], sds['beta2'])])

        try:
            pvals.to_hdf(os.path.join(helpers.resultsdir, 'meg_hierarchical_advi.h5'), 'pvals', mode='w')
        except:
            pass