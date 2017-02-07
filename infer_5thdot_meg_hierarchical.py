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
from scipy.stats import norm, ttest_1samp
import os
import statsmodels.api as sm


#%% 
# one of 'advi', 'nuts', 'ss'
method = 'ss'

fname = 'meg_hierarchical_' + method + '.h5'

doti = 5

# implements the assumption that in a trial with RT=(dot onset time + rt_thresh)
# there cannot be an effect of the corresponding dot in the MEG signal
rt_thresh = 0.1


#%% load all data
epochs_all = helpers.load_meg_epochs(hfreq=10, window=[0.4, 0.7])
subjects = epochs_all.index.levels[0]
S = subjects.size

times = epochs_all.index.levels[2]


#%% load trial-level design matrices for all subjects (ith-dot based)
DM = subject_DM.get_ithdot_DM(doti=doti, r_names=['dot_y', 'surprise', 
        'logpost_left', 'entropy', 'trial_time', 'intercept', 'RT'])
DM = DM.loc(axis=0)[subjects, :]

# remove trials which had too few simulations to estimate surprise
good_trials = np.logical_not(np.isnan(DM['surprise'])).values
                         
# remove trials which had RTs below a threshold (and therefore most likely 
# cannot have an effect of the shown dot)
good_trials = np.logical_and(good_trials, 
                             DM['RT'] >= helpers.dotdt*(doti-1) + rt_thresh)
# remove RT from design matrix
del DM['RT']

# final number of regressors
R = DM.shape[1]

# select only the good trials
DM = DM.loc[good_trials]

sub_map = {x: np.flatnonzero(subjects==x)[0] for x in subjects}
subject_idx = DM.index.get_level_values('subject').map(lambda x: sub_map[x])


#%% prepare output and helpers
second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([epochs_all.columns, 
              epochs_all.index.levels[2]], names=['channel', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

if method == 'advi':
    # 2nd-level posterior probabilities
    def post_prob(mean, sd):
        if mean > 0:
            return norm.cdf(0, mean, sd)
        else:
            return 1 - norm.cdf(0, mean, sd)


#%% infer
for time in times:
    print('time = %d' % time)
    for channel in epochs_all.columns:
        data = epochs_all.loc[(slice(None), slice(None), time), channel]
        data = data.loc[good_trials.values]
            
        if method == 'ss':
            params = np.zeros((S, R))
            for s, sub in enumerate(subjects):
                res = sm.OLS(data.loc[sub].values, DM.loc[sub].values, 
                             hasconst=True).fit()
                params[s, :] = res.params
               
            second_level.loc[(channel, time), ('mean', slice(None))] = (
                    params.mean(axis=0))
            second_level.loc[(channel, time), ('std', slice(None))] = (
                    params.std(axis=0))
            _, pvals = ttest_1samp(params, 0, axis=0)
            second_level.loc[(channel, time), ('mlog10p', slice(None))] = (
                    -np.log10(pvals))
        else:
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
                second_level.to_hdf(os.path.join(helpers.resultsdir, fname), 
                                    'second_level', mode='w')
            except:
                pass
    
    # for summary statistic only save every time point
    if method == 'ss':
        try:
            second_level.to_hdf(os.path.join(helpers.resultsdir, fname), 
                                'second_level', mode='w')
        except:
            pass
    else:
        # print new line for pyMC3-methods
        print('')