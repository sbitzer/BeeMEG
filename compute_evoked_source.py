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
from warnings import warn
import source_statistics as ss
import regressors


#%% options and load data
# source data
src_fname = 'source_epochs_allsubs_HCPMMP1_5_8_201802081827.h5'
src_path = os.path.join(ss.bem_dir, src_fname)

nperm = 3

normdata = 'trials'

response_aligned = True

exclude_to = True

timeslice = [-1000, 1000]


#%% create output file
file = pd.datetime.now().strftime('evoked_source'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [response_aligned, exclude_to], 
            ['response_aligned', 'exclude_to'])
    store['normdata'] = pd.Series(normdata)
    store['srcfile'] = pd.Series(src_path)


#%% get info about source data
with pd.HDFStore(src_path, 'r') as store:
    epochs = store.select('label_tc', 'subject==2')
    
    if normdata == 'global':
        epochs_mean = store.epochs_mean
        epochs_std = store.epochs_std
    
    gc.collect()

index = epochs.loc[2].index
epochtimes = index.levels[1]
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

datalabels = epochs.columns


#%% reindex time if necessary and select desired time slice
rts = regressors.subject_trial.RT.loc[subjects] * 1000
choices = regressors.subject_trial.response.loc[subjects]

# if RTs are an exact multiple of 5, numpy will round both 10+5 and 20+5 to
# 20; so to not get duplicate times add a small jitter
fives = (np.mod(rts, 10) != 0) & (np.mod(rts, 5) == 0)
# subtracting the small amount means that the new times will be rounded
# down so that e.g., -5 becomes 0, I prefer that round because then there
# might be slightly more trials with larger times, but the effect is small
rts[fives] = rts[fives] - 1e-7

if response_aligned:
    def rtindex(sub):
        """Returns index for data of a subject with time as difference to RT"""
        
        rttime = (  epochtimes.values[None, :] 
                  - rts.loc[sub].values[:, None]).flatten()
        # round to 10 ms and convert into integer
        rttime = rttime.round(-1).astype(int)
    
        # create new subject-trial-time index (I cannot only set the labels of
        # the time level, because the response aligned times have different 
        # values than those stored in dot-onset aligned times)
        return pd.MultiIndex.from_arrays(
                [np.full_like(rttime, sub),
                 np.tile(np.arange(1, 481)[:, None], 
                         (1, epochtimes.size)).flatten(),
                 rttime], names=['subject', 'trial', 'time']).sort_values()
    
    if len(timeslice) != 2:
        raise ValueError("For response aligned times you need to provide the "
                         "start and stop of the time window!")
    times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                     name='time')
else:
    times = epochtimes[slice(*epochtimes.slice_locs(*timeslice))]


#%% prepare output stores
first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), datalabels, times], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['mean'], ['evoked']], 
              names=['subject', 'measure', 'regressor']), dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), datalabels, times], 
              names=['permnr', 'label', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'tval', 'mlog10p'], 
              ['evoked']], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)


#%% run inference
# original sequential order
permutation = np.arange(480 * epochtimes.size).reshape([480, epochtimes.size])

for perm in np.arange(nperm+1):
    print('\npermutation %d' % perm)
    print('-------------')
    
    if perm > 0:
        # randomly permute time points, use different permutations for trials, 
        # but use the same permutation across all subjects such that only times
        # are permuted, but random associations between subjects are maintained
        for tr in range(480):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[tr, :])
    
    for sub in subjects:
        print('\rsubject = %2d' % sub, end='', flush=True)
        
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
        
        # change to response-aligned times, when desired
        if response_aligned:
            epochs.index = rtindex(sub)
            
        # exclude timed out trials from analysis, if desired
        if exclude_to:
            ch_sub = choices.loc[sub]
            to = ch_sub[ch_sub == helpers.toresponse[0]].index.values
            
            # nan-ing the first column only is sufficient when later dropping
            # all rows with at least one nan
            epochs.loc[(sub, to, slice(None)), epochs.columns[0]] = np.nan

        # permute
        epochs.loc[:] = epochs.values[permutation.flatten(), :]
        
        # drop nan-ed data
        epochs.dropna(inplace=True)
        
        # select desired times
        if len(timeslice) > 0:
            epochs = epochs.loc[(slice(None), slice(None), times), :]
        
        # compute average across trials
        first_level.loc[(perm, slice(None)), sub] = epochs.groupby(
                'time').mean().T.stack().values
        
    
    print('\ncomputing second level ...')
    for label in datalabels:
        params = first_level.loc[(perm, label, slice(None)), :]
        
        second_level.loc[(perm, label, slice(None)), ('mean', 'evoked')] = (
                params.mean(axis=1).values)
        second_level.loc[(perm, label, slice(None)), ('std', 'evoked')] = (
                params.std(axis=1).values)
        tvals, pvals = ttest_1samp(params.values, 0, axis=1)
        second_level.loc[(perm, label, slice(None)), ('tval', 'evoked')] = (
            tvals)
        second_level.loc[(perm, label, slice(None)), ('mlog10p', 'evoked')] = (
                -np.log10(pvals))
        
    try:
        with pd.HDFStore(tmpfile, mode='w') as store:
            store['second_level'] = second_level
            store['first_level'] = first_level
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
except:
    warn('Results not saved yet!')
else:
    os.remove(tmpfile)