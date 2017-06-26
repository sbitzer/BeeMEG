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
from warnings import warn


#%% options
# which dot to investigate?
dots = np.arange(1, 6)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = 0.1

# names of regressors that should enter the GLM
r_names = ['abs_dot_x', 'abs_dot_y', 'dot_x', 'dot_y', 'entropy', 'trial_time', 
           'intercept', 'response']
R = len(r_names)

# source data to use
srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)

# How many permutations should be computed?
nperm = 3

# where to store
file = pd.datetime.now().strftime('source_singledot'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series([rt_thresh], ['rt_thresh'])
    store['dots'] = pd.Series(dots)
    store['srcfile'] = pd.Series(srcfile)


#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

times = epochs.index.levels[2]


#%% load trial-level design matrices for all subjects (ith-dot based)
if 'RT' in r_names:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names)
else:
    DM = subject_DM.get_trial_DM(dots, r_names=r_names+['RT'])

# move_dist_1 is constant at 0, we don't need to fit that
if 'move_dist' in r_names:
    del DM['move_dist_1']
    
DM = DM.loc(axis=0)[subjects, :]

# remove trials which had RTs below a threshold (and therefore most likely 
# cannot have an effect of the last considered dot)
good_trials = DM['RT'] >= (helpers.dotdt*(dots.max()-1) + rt_thresh)
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

# just get mean and std of data across all subjects, trials and times
epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% prepare output and helpers
srclabels = epochs_mean.index

first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              DM.columns], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times], 
              names=['permnr', 'label', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% infer

# original sequential order
permutation = np.arange(480 * times.size)

for perm in np.arange(nperm+1):
    print('\npermutation %d' % perm)
    print('-------------')
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        permutation = np.arange(480 * times.size).reshape([480, times.size])
        for t in range(times.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, t])
    
    with pd.HDFStore(srcfile, 'r') as store:
        for s, sub in enumerate(subjects):
            epochs = store.select('label_tc', 'subject=sub')
            epochs.loc[:] = epochs.values[permutation.flatten(), :]
            
            for t0 in times:
                print('\rsubject = %2d, t0 = %3d' % (sub, t0), end='')
        
                data = epochs.loc[(sub, slice(None), t0)]
                data = data.loc[good_trials.loc[sub].values]
        
                for label in srclabels:
                    res = sm.OLS(data[label].values, DM.loc[sub].values, 
                                 hasconst=True).fit()
                
                    first_level.loc[(perm, label, t0), (sub, 'beta', slice(None))] = (
                        res.params)
                    first_level.loc[(perm, label, t0), (sub, 'bse', slice(None))] = (
                        res.bse)
                    
                    first_level_diagnostics.loc[(perm, label, t0), (sub, 'Fval')] = (
                        res.fvalue)
                    first_level_diagnostics.loc[(perm, label, t0), (sub, 'R2')] = (
                        res.rsquared)
                    first_level_diagnostics.loc[(perm, label, t0), (sub, 'llf')] = (
                        res.llf)
    
    print('\ncomputing second level ...') 
    for label in srclabels:
        for t0 in times:
            params = first_level.loc[(perm, label, t0), 
                                     (slice(None), 'beta', slice(None))]
            params = params.reset_index(level='regressor').pivot(columns='regressor')
            
            second_level.loc[(perm, label, t0), ('mean', slice(None))] = (
                    params.mean().values)
            second_level.loc[(perm, label, t0), ('std', slice(None))] = (
                    params.std().values)
            _, pvals = ttest_1samp(params.values, 0, axis=0)
            second_level.loc[(perm, label, t0), ('mlog10p', slice(None))] = (
                    -np.log10(pvals))
            
    try:
        with pd.HDFStore(tmpfile, mode='w') as store:
            store['second_level'] = second_level
            store['first_level'] = first_level
            store['first_level_diagnostics'] = first_level_diagnostics
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
        store['first_level_diagnostics'] = first_level_diagnostics
except:
    warn('Results not saved yet!')
else:
    os.remove(tmpfile)
    
