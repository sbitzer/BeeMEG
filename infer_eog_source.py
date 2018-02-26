#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:02:35 2018

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
from scipy.stats import ttest_1samp
import os, gc
import statsmodels.api as sm
from warnings import warn
import source_statistics as ss


#%% options and load data
# horizontal EOG events
eog_fname = 'eog_201802081218.h5'
eog_path = os.path.join(helpers.resultsdir, eog_fname)

# source data
src_fname = 'source_epochs_allsubs_HCPMMP1_5_8_201802081827.h5'
src_path = os.path.join(ss.bem_dir, src_fname)

eog_event_limits = np.r_[-100, 50]

nperm = 3

normdata = 'trials'


#%% create output file
file = pd.datetime.now().strftime('eog_source'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['eog_event_limits'] = pd.Series(eog_event_limits)
    store['normdata'] = pd.Series(normdata)
    store['eogfile'] = pd.Series(eog_path)
    store['srcfile'] = pd.Series(src_path)


#%% get info about source data
with pd.HDFStore(src_path, 'r') as store:
    epochs = store.select('label_tc', 'subject==2')
    
    if normdata == 'global':
        epochs_mean = store.epochs_mean
        epochs_std = store.epochs_std
    
    gc.collect()

index = epochs.loc[2].index
times = index.levels[1]
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

datalabels = epochs.columns


#%% create EOG event regressor
eog_events = pd.read_hdf(eog_path, 'eog_events')

def create_eog_DM(sub, evlims):
    evlims = np.array(evlims)
    
    eogDM = pd.DataFrame(np.c_[np.zeros(index.size), np.ones(index.size)], 
                         index=index, columns=['eog_events', 'intercept'])
    eogDM.sort_index(axis=1, inplace=True)
    
    for trial, time in eog_events.loc[sub].index:
        eogDM.loc[(trial, slice(*np.round(time + evlims, -1))), 
                  'eog_events'] = 1
        
    return eogDM

regnames = create_eog_DM(2, eog_event_limits).columns


#%% prepare output stores
first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), datalabels], 
              names=['permnr', 'label']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              regnames], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), datalabels], 
              names=['permnr', 'label']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), datalabels], 
              names=['permnr', 'label']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'tval', 'mlog10p'], 
              regnames], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == regnames), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == regnames), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% run inference
# original sequential order
permutation = np.arange(index.size).reshape([480, times.size])

for perm in np.arange(nperm+1):
    print('\npermutation %d' % perm)
    print('-------------')
    
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        for t in range(times.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, t])
    
    for sub in subjects:
        print('\rsubject = %2d' % sub, end='', flush=True)
        
        # create DM
        DM = create_eog_DM(sub, eog_event_limits)
        
        # load data
        with pd.HDFStore(src_path, 'r') as store:
            epochs = store.select('label_tc', 'subject==sub')
            gc.collect()
            
        # normalise
        if normdata == 'global':
            epochs = (epochs - epochs_mean) / epochs_std
        elif normdata == 'trials':
            epochs = epochs.unstack('time')
            epochs = (epochs - epochs.mean()) / epochs.std()
            epochs = epochs.stack('time')
        
        # permute
        epochs.loc[:] = epochs.values[permutation.flatten(), :]
    
        # run GLM
        for label in datalabels:
            # data are log-transformed absolute values, because I don't know the direction
            # of the effects and this just tests for an effect in the magnitude
            data = np.log(epochs[label].abs().values)
            
            res = sm.OLS(data, DM.values, hasconst=True).fit()
    
            first_level.loc[(perm, label), 
                            (sub, 'beta', slice(None))] = res.params
            first_level.loc[(perm, label), 
                            (sub, 'bse', slice(None))] = res.bse
                        
            first_level_diagnostics.loc[(perm, label), 
                                        (sub, 'Fval')] = res.fvalue
            first_level_diagnostics.loc[(perm, label), 
                                        (sub, 'R2')] = res.rsquared
            first_level_diagnostics.loc[(perm, label), 
                                        (sub, 'llf')] = res.llf
    
    print('\ncomputing second level ...')
    for label in datalabels:
        params = first_level.loc[(perm, label), 
                                 (slice(None), 'beta', slice(None))]
        params = params.unstack('regressor')
        
        second_level.loc[(perm, label), ('mean', slice(None))] = (
                params.mean().values)
        second_level.loc[(perm, label), ('std', slice(None))] = (
                params.std().values)
        tvals, pvals = ttest_1samp(params.values, 0, axis=0)
        second_level.loc[(perm, label), ('tval', slice(None))] = (
            tvals)
        second_level.loc[(perm, label), ('mlog10p', slice(None))] = (
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