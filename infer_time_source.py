#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:55:57 2017

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import subject_DM
from scipy.stats import ttest_1samp
import os
import statsmodels.api as sm
import mne
from warnings import warn
import gc


#%% options
# names of regressors that should enter the GLM
r_names = ['accev_time', 'motoprep', 'dotcount', 'motoresponse']
R = len(r_names)
# sort for use in index below
r_names.sort()

# label mode = mean
srcfile = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'

# label mode = mean_flip
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'

# label mode= None, lselection=[V1, V2], window=[0, 0.4]
#srcfile = 'source_epochs_allsubs_HCPMMP1_201708151431.h5'

# label mode = (abs) max
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706271717.h5'

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)

# How many permutations should be computed?
nperm = 3

# whether to normalise the source signal to have mean 0 and std 1 in each area
# across trials, time points and subjects
normsrc = True

accev_delay = 0.4

# where to store
file = pd.datetime.now().strftime('source_time'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series([normsrc, accev_delay], 
         ['normsrc', 'accev_delay'])
    store['srcfile'] = pd.Series(srcfile)
    
    
#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

times = epochs.index.levels[2]


#%% create design matrix
print('creating design matrix ...', flush=True)
regfuns = subject_DM.regressors.subject_trial_time.copy()
regfuns['accev_time'] = (
        lambda trt: subject_DM.regressors.subject_trial_time['accev_time']
                    (trt, accev_delay))

DM = pd.concat([regfuns[r_name](times / 1000) for r_name in r_names], 
               axis=1, keys=r_names)

DM = subject_DM.normalise_DM(DM, 'subject').sort_index()
    
# add intercept
DM['intercept'] = 1
DM.sort_index(axis=1, inplace=True)


#%% check condition numbers
DM_condition_numbers = subject_DM.compute_condition_number(DM, scope='subject')

with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
    store['DM_condition_numbers'] = DM_condition_numbers

# warn, if there are more than 2 subjects with condition numbers above 10 at
# any time point from 100 ms within the trial
if np.any((DM_condition_numbers > 10).sum() > 2):
    warn("High condition numbers of design matrix detected!")


#%% get mean and std of data across all subjects, trials and times
epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')
srclabels = epochs_mean.index


#%% prepare output stores
first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels], 
              names=['permnr', 'label']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              DM.columns], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels], 
              names=['permnr', 'label']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels], 
              names=['permnr', 'label']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'tval', 'mlog10p', 
                                             'significant', 'mlog10pcorr'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% run analysis

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
            
    for s, sub in enumerate(subjects):
        print('\rsubject = %2d' % sub, end='', flush=True)
            
        with pd.HDFStore(srcfile, 'r') as store:
            epochs = store.select('label_tc', 'subject=sub').copy()
        
        # normalise each label across time points and subjects
        if normsrc:
            epochs = (epochs - epochs_mean) / epochs_std
        
        epochs.loc[:] = epochs.values[permutation.flatten(), :]
        
        data = epochs.loc[sub]
        
        for label in srclabels:
            res = sm.OLS(data[label].values, DM.loc[sub].values).fit()
        
            first_level.loc[(perm, label), (sub, 'beta', slice(None))] = (
                res.params)
            first_level.loc[(perm, label), (sub, 'bse', slice(None))] = (
                res.bse)
            
            first_level_diagnostics.loc[(perm, label), (sub, 'Fval')] = (
                res.fvalue)
            first_level_diagnostics.loc[(perm, label), (sub, 'R2')] = (
                res.rsquared)
            first_level_diagnostics.loc[(perm, label), (sub, 'llf')] = (
                res.llf)

        # garbage collect every 5 subjects, for some reason I need to call this
        # manually as otherwise memory usage increases heavily through 
        # iterations of loading data for individual subjects
        if np.mod(s, 5) == 0:
            gc.collect()
                
                
    print('\ncomputing second level ...')
    for label in srclabels:
        params = first_level.loc[(perm, label), 
                                 (slice(None), 'beta', slice(None))]
        params = params.reset_index(level='regressor').pivot(columns='regressor')
        
        second_level.loc[(perm, label), ('mean', slice(None))] = (
                params.mean().values)
        second_level.loc[(perm, label), ('std', slice(None))] = (
                params.std().values)
        tvals, pvals = ttest_1samp(params.values, 0, axis=0)
        second_level.loc[(perm, label), ('tval', slice(None))] = (
                tvals)
        second_level.loc[(perm, label), ('mlog10p', slice(None))] = (
                -np.log10(pvals))
        
    # FDR-correct p-values across all regressors of interest (ignore intercept)
    rej, pcorr = mne.stats.fdr_correction(
            10 ** -second_level.loc[perm, ('mlog10p', r_names)].values.flatten())
    second_level.loc[perm, ('significant', r_names)] = rej.reshape(
            srclabels.size, R)
    second_level.loc[perm, ('mlog10pcorr', r_names)] = -np.log10(pcorr).reshape(
            srclabels.size, R)

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
    