#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:37:21 2017

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import subject_DM
from scipy.stats import ttest_1samp
import os
import statsmodels.api as sm


#%% options
# which dots to investigate?
dots = np.arange(1, 6)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = 0.1

# names of regressors that should enter the GLM
r_names = ['abs_dot_y', 'abs_dot_x', 'dot_y', 'dot_x', 'entropy', 'trial_time', 
           'intercept', 'response', 'dot_x_cflip']

# baseline period (set to None for no baseline correction)
bl = (-0.3, 0)

# desired sampling frequency of data
sfreq = 100

# smoothing frequency; the lower, the more smoothing; set to sfreq for no 
# smoothing, otherwise can be maximally sfreq/2
hfreq = sfreq

# time window within an epoch to consider
window = [0, 0.9]

# How many permutations should be computed?
nperm = 5

# where to store
file = pd.datetime.now().strftime('meg_singledot'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join(os.path.expanduser('~'), file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='zlib') as store:
    store['scalar_params'] = pd.Series([rt_thresh, sfreq, hfreq], 
         ['rt_thresh', 'sfreq', 'hfreq'])
    store['dots'] = pd.Series(dots)
    store['bl'] = pd.Series(bl)
    store['window'] = pd.Series(window)


#%% load all data
epochs_all = helpers.load_meg_epochs(hfreq, sfreq, window, bl=bl)
subjects = epochs_all.index.levels[0]
S = subjects.size

times = epochs_all.index.levels[2]


#%% load trial-level design matrices for all subjects (ith-dot based)
if 'RT' in r_names:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names)
else:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names+['RT'])
    
DM = DM.loc(axis=0)[subjects, :]

# remove trials which had too few simulations to estimate surprise
good_trials = np.ones(DM.shape[0], dtype=bool)
if 'accev' in r_names:
    snames = ['accev_%d' % d for d in dots]
    good_trials = np.logical_not(np.any(np.isnan(DM[snames]), axis=1))
if 'accsur_pca_%d' % dots[0] in DM.columns:
    snames = ['accsur_pca_%d' % d for d in dots]
    good_trials = np.logical_and(good_trials, 
            np.logical_not(np.any(np.isnan(DM[snames]), axis=1)))
                         
# remove trials which had RTs below a threshold (and therefore most likely 
# cannot have an effect of the last considered dot)
good_trials = np.logical_and(good_trials, 
                             DM['RT'] >= helpers.dotdt*(dots.max()-1) + rt_thresh)
# remove RT from design matrix
if 'RT' not in r_names:
    del DM['RT']

# select only the good trials
DM = DM.loc[good_trials]

# sort regressor names (columns) for indexing
DM.sort_index(axis=1, inplace=True)
R = DM.shape[1]


#%% normalise data and regressors to simplify definition of priors
DM = (DM - DM.mean()) / DM.std()
# intercept will be nan, because it has no variance
DM['intercept'] = 1

# this will include bad trials, too, but it shouldn't matter as this 
# normalisation across subjects, trials and time points is anyway very rough
# do this in two steps to save some memory
with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
    store['epochs_mean'] = epochs_all.mean()
epochs_all = epochs_all - epochs_all.mean()
with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
    store['epochs_std'] = epochs_all.std()
epochs_all = epochs_all / epochs_all.std()


#%% prepare output and helpers
first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              epochs_all.columns, times], 
              names=['permnr', 'channel', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              DM.columns], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              epochs_all.columns, times], 
              names=['permnr', 'channel', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              epochs_all.columns, times], 
              names=['permnr', 'channel', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% infer
for perm in np.arange(nperm+1):
    print('permutation %d' % perm)
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        permutation = np.arange(epochs_all.shape[0]).reshape([x.size for x in epochs_all.index.levels])
        for t in range(times.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, :, t].T)
        epochs_all.loc[:] = epochs_all.values[permutation.flatten(), :]
        
    for t0 in times:
        print('t0 = %d' % t0)
        
        for channel in epochs_all.columns:
            data = epochs_all.loc[(slice(None), slice(None), t0), channel]
            data = data.loc[good_trials.values]
                
            params = np.zeros((S, R))
            for s, sub in enumerate(subjects):
                res = sm.OLS(data.loc[sub].values, DM.loc[sub].values, 
                             hasconst=True).fit()
                params[s, :] = res.params
                
                first_level.loc[(perm, channel, t0), (sub, 'beta', slice(None))] = (
                    params[s, :])
                first_level.loc[(perm, channel, t0), (sub, 'bse', slice(None))] = (
                    res.bse)
                
                first_level_diagnostics.loc[(perm, channel, t0), (sub, 'Fval')] = (
                    res.fvalue)
                first_level_diagnostics.loc[(perm, channel, t0), (sub, 'R2')] = (
                    res.rsquared)
                first_level_diagnostics.loc[(perm, channel, t0), (sub, 'llf')] = (
                    res.llf)
               
            second_level.loc[(perm, channel, t0), ('mean', slice(None))] = (
                    params.mean(axis=0))
            second_level.loc[(perm, channel, t0), ('std', slice(None))] = (
                    params.std(axis=0))
            _, pvals = ttest_1samp(params, 0, axis=0)
            second_level.loc[(perm, channel, t0), ('mlog10p', slice(None))] = (
                    -np.log10(pvals))
            
    try:
        with pd.HDFStore(tmpfile, mode='w') as store:
            store['second_level'] = second_level
            store['first_level'] = first_level
            store['first_level_diagnostics'] = first_level_diagnostics
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
        store['first_level_diagnostics'] = first_level_diagnostics
except:
    pass
else:
    os.remove(tmpfile)