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


#%% options
# one of 'advi', 'nuts', 'ss'
method = 'ss'

# where to store
file = os.path.join(helpers.resultsdir, 'meg_hierarchical_' + method + '_bl_long_1to5_permuted2.h5')

# which dot to investigate?
dots = np.arange(1, 6)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the corresponding dot in the MEG signal
rt_thresh = 0.1

# names of regressors that should enter the GLM
r_names = ['dot_y', 'surprise', 'dot_x', 'entropy', 'trial_time', 
           'intercept']

# number of ADVI iterations
advi_n = 30000

# priors of Bayesian hierarchical model
prior = pd.Series({'beta2_sd': 100, 
                   'sigma2_beta': 5, 
                   'eps_beta': 5})

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='zlib') as store:
    store['dots'] = pd.Series(dots)
    store['rt_thresh'] = pd.Series(rt_thresh)
    if method in ['advi', 'nuts']:
        store['prior'] = prior
    if method == 'advi':
        store['advi_n'] = pd.Series(advi_n)
         

#%% load all data
#window = doti * helpers.dotdt + np.array([-0.1, 0.2])
window = [0, 0.9]
epochs_all = helpers.load_meg_epochs(hfreq=100, window=window, bl=(-0.3, 0))
subjects = epochs_all.index.levels[0]
S = subjects.size

times = epochs_all.index.levels[2]

# randomly permute trials, use different permutations for time points, but use
# the same permutation across all subjects such that only trials are permuted,
# but random associations between subjects are maintained
perm = np.arange(epochs_all.shape[0]).reshape([x.size for x in epochs_all.index.levels])
for t in range(times.size):
    # shuffles axis=0 inplace
    np.random.shuffle(perm[:, :, t].T)
epochs_all.loc[:] = epochs_all.values[perm.flatten(), :]


#%% load trial-level design matrices for all subjects (ith-dot based)
if 'RT' in r_names:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names)
else:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names+['RT'])
    
DM = DM.loc(axis=0)[subjects, :]

# ensure that order of regressors in design matrix and results DataFrame 
# (second_level) is the same
DM.sort_index(axis=1, inplace=True)

# remove trials which had too few simulations to estimate surprise
snames = ['surprise_%d' % d for d in dots]
good_trials = np.logical_not(np.any(np.isnan(DM[snames]), axis=1))
                         
# remove trials which had RTs below a threshold (and therefore most likely 
# cannot have an effect of the last considered dot)
good_trials = np.logical_and(good_trials, 
                             DM['RT'] >= helpers.dotdt*(dots.max()-1) + rt_thresh)
# remove RT from design matrix
if 'RT' not in r_names:
    del DM['RT']

# final number of regressors
R = DM.shape[1]

# select only the good trials
DM = DM.loc[good_trials]

sub_map = {x: np.flatnonzero(subjects==x)[0] for x in subjects}
subject_idx = DM.index.get_level_values('subject').map(lambda x: sub_map[x])


#%% normalise data and regressors to simplify definition of priors
DM = (DM - DM.mean()) / DM.std()
# intercept will be nan, because it has no variance
DM['intercept'] = 1

# this will include bad trials, too, but it shouldn't matter as this 
# normalisation across subjects, trials and time points is anyway very rough
epochs_all = (epochs_all - epochs_all.mean()) / epochs_all.std()


#%% prepare output and helpers
second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([epochs_all.columns, 
              epochs_all.index.levels[2]], names=['channel', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'

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
                # second-level parameter values
                beta2 = pm.Normal('beta2', sd=prior.beta2_sd**2, shape=R)
                
                # between-subject std
                sigma2 = pm.HalfCauchy('sigma2', prior.sigma2_beta, shape=R)
                
                # subject-specific parameter values
                beta1 = pm.Normal('beta1', mu=beta2, sd=sigma2, shape=(S, R))
                
                # within-subject std
                eps = pm.HalfCauchy('eps', prior.eps_beta)
                
                pred = (beta1[subject_idx, :] * DM.values).sum(axis=1)
                
                likelihood = pm.Normal(channel, mu=pred, sd=eps, observed=data.values)
                
                means, sds, elbos = pm.variational.advi(n=advi_n)
            
            # compute "posterior p-values"
            mlog10p = -np.log10(np.array([post_prob(m, s) for m, s in 
                                          zip(means['beta2'], sds['beta2'])]))
    
            second_level.loc[(channel, time), ('mean', slice(None))] = (
                    means['beta2'])
            second_level.loc[(channel, time), ('std', slice(None))] = (
                    sds['beta2'])
            second_level.loc[(channel, time), ('mlog10p', slice(None))] = (
                    mlog10p)
    
            try:
                with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
                    store['second_level'] = second_level
            except:
                pass
    
    # for summary statistic save only every time point
    if method == 'ss':
        try:
            with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
                store['second_level'] = second_level
        except:
            pass
    else:
        # print new line for pyMC3-methods
        print('')