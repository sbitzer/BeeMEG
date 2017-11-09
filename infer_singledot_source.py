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
import re
import statsmodels.api as sm
from warnings import warn


#%% options
# which dot to investigate?
dots = np.arange(5, 8)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = -1.0

# names of regressors that should enter the GLM
r_names = ['entropy', 'trial_time', 'intercept', 'response']
R = len(r_names)

# source data to use

# label mode = mean
#srcfile = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'

# label mode = mean_flip
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'

# label mode= None, lselection=[V1, V2], window=[0, 0.4]
#srcfile = 'source_epochs_allsubs_HCPMMP1_201708151431.h5'

# label mode = (abs) max
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706271717.h5'

# label mode= None, lselection=pre-motor and motor areas, window=[0.3, 0.9]
srcfile = 'source_epochs_allsubs_HCPMMP1_201710231731.h5'

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)

# whether a response-aligned analysis should be run, i.e., whether time should
# be defined with respect to the response time; if True, set timeslice below
# providing both ends
response_aligned = True

# time window for which to run the analysis, must be a 0-, 1- or 2-element list
# if empty list, all times in srcfile are used for analysis;
# if 1-element, all times starting with the given will be included in analysis;
# if 2-element, these are the slice limits considered (right edge is inclusive)
# note that normalisation of data is still over all times in srcfile
timeslice = [-200, 50]

# How many permutations should be computed?
nperm = 3

# whether to normalise the source signal to have mean 0 and std 1 in each area
# across trials, time points and subjects
normsrc = True

# where to store
file = pd.datetime.now().strftime('source_singledot'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series([rt_thresh, float(normsrc), 
         float(response_aligned)], ['rt_thresh', 'normsrc', 'response_aligned'])
    store['dots'] = pd.Series(dots)
    store['srcfile'] = pd.Series(srcfile)


#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

times = epochs.index.levels[2]


#%% reindex time if necessary and select desired time slice
if response_aligned:
    epochtimes = epochs.index.levels[2]
    rts = subject_DM.regressors.subject_trial.RT.loc[subjects] * 1000
    
    # if RTs are an exact multiple of 5, numpy will round both 10+5 and 20+5 to
    # 20; so to not get duplicate times add a small jitter
    fives = (np.mod(rts, 10) != 0) & (np.mod(rts, 5) == 0)
    # subtracting the small amount means that the new times will be rounded
    # down so that e.g., -5 becomes 0, I prefer that round because then there
    # might be slightly more trials with larger times, but the effect is small
    rts[fives] = rts[fives] - 1e-7
    
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
    times = times[slice(*times.slice_locs(*timeslice))]


#%% load trial-level design matrices for all subjects (ith-dot based)
if 'RT' in r_names:
    DM = subject_DM.get_trial_DM(dots, subjects, r_names=r_names)
else:
    DM = subject_DM.get_trial_DM(dots, subjects, r_names=r_names+['RT'])

# add selected single dot regressors
single_r_names = []
single_r_names_components = []
for r_name in r_names:
    match = re.match(r'(.*)_(\d)$', r_name)
    if match:
        single_r_names.append(r_name)
        single_r_names_components.append(match.groups())

DM = pd.concat([DM]+[subject_DM.get_trial_DM(int(dot), subjects, r_names=[r_name]) 
                     for r_name, dot in single_r_names_components],
               axis=1)

# move_dist_1 is constant at 0, we don't need to fit that
if 'move_dist' in r_names:
    del DM['move_dist_1']
    
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


#%% normalise design matrix to simplify definition of priors
DM = subject_DM.normalise_DM(DM, 'subject')

if DM.isnull().any().any():
    raise ValueError("Design matrix has NaNs somewhere, please check!")

# just get mean and std of data across all subjects, trials and times
epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% check collinearity in design matrix using condition number
# numbers from 10 indicate some problem with collinearity in design matrix
# (see Rawlings1998, p. 371)
DM_condition_numbers = subject_DM.compute_condition_number(DM, scope='subject')

with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
    store['DM_condition_numbers'] = DM_condition_numbers

# warn, if there are more than 2 subjects with condition numbers above 10 at
# any time point from 100 ms within the trial
if (DM_condition_numbers > 10).sum() > 2:
    warn("High condition numbers of design matrix detected!")


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
        columns=pd.MultiIndex.from_product([['mean', 'std', 'tval', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'

if response_aligned:
    trial_counts = pd.DataFrame([], index=times, 
                                columns=pd.Index(subjects, name='subject'),
                                dtype=int)


#%% infer
for perm in np.arange(nperm+1):
    print('\npermutation %d' % perm)
    print('-------------')

    # original sequential order
    permutation = np.arange(480 * times.size).reshape((480, times.size))
    
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        for t in range(times.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, t])
    
    with pd.HDFStore(srcfile, 'r') as store:
        for s, sub in enumerate(subjects):
            epochs = store.select('label_tc', 'subject=sub')
            if response_aligned:
                epochs.index = rtindex(sub)
                
            if len(timeslice) > 0:
                epochs = epochs.loc[(slice(None), slice(None), times), :]
            
            if response_aligned:
                if perm == 0:
                    trial_counts[sub] = epochs.iloc[:, 0].groupby(
                            level='time').count()
                
                # shuffling needs more effort here as not all trials are 
                # present at each time point, the strategy is to make a full
                # data array with nans where there is no data, shuffle
                # trials according to the generated shuffling order and then
                # drop the nans again
                epochs_full = pd.DataFrame([], 
                        index=pd.MultiIndex.from_product(
                                [[sub], np.arange(1, 481), times], 
                                names=['subject', 'trial', 'time']), 
                        columns=epochs.columns, 
                        dtype=float)
                
                # fill the array with data from each time point
                for ti, t0 in enumerate(times):
                    trind = list(epochs.xs(t0, level='time')
                                 .index.get_level_values('trial'))
                    epochs_full.loc[(sub, trind, t0), :] = epochs.xs(
                            t0, level='time').values
                
                # perform the shuffling
                epochs_full.loc[:] = epochs_full.values[
                        permutation.flatten(), :]
                
                epochs = epochs_full.dropna()
                del epochs_full
            else:
                epochs.loc[:] = epochs.values[permutation.flatten(), :]
            
            # normalise each label across time points and subjects
            if normsrc:
                epochs = (epochs - epochs_mean) / epochs_std
            
            for t0 in times:
                print('\rsubject = %2d, t0 = % 4d' % (sub, t0), 
                      end='', flush=True)
        
                data = epochs.loc[(sub, slice(None), t0)]
                data = data.loc[good_trials.loc[sub].loc[data.index].values]
                
                if data.size > 0:
                    for label in srclabels:
                        res = sm.OLS(data[label].values, 
                                     DM.loc[(sub, list(data.index)), :].values, 
                                     hasconst=True).fit()
                    
                        first_level.loc[(perm, label, t0), 
                                        (sub, 'beta', slice(None))] = (
                            res.params)
                        first_level.loc[(perm, label, t0), 
                                        (sub, 'bse', slice(None))] = (
                            res.bse)
                        
                        first_level_diagnostics.loc[(perm, label, t0), 
                                                    (sub, 'Fval')] = (
                            res.fvalue)
                        first_level_diagnostics.loc[(perm, label, t0), 
                                                    (sub, 'R2')] = (
                            res.rsquared)
                        first_level_diagnostics.loc[(perm, label, t0), 
                                                    (sub, 'llf')] = (
                            res.llf)
    
    print('\ncomputing second level ...') 
    for label in srclabels:
        for t0 in times:
            params = first_level.loc[(perm, label, t0), 
                                     (slice(None), 'beta', slice(None))]
            params = params.reset_index(level='regressor').pivot(
                    columns='regressor')
            
            second_level.loc[(perm, label, t0), ('mean', slice(None))] = (
                    params.mean().values)
            second_level.loc[(perm, label, t0), ('std', slice(None))] = (
                    params.std().values)
            tvals, pvals = ttest_1samp(params.values, 0, axis=0)
            second_level.loc[(perm, label, t0), ('tval', slice(None))] = (
                tvals)
            second_level.loc[(perm, label, t0), ('mlog10p', slice(None))] = (
                    -np.log10(pvals))
            
    try:
        with pd.HDFStore(tmpfile, mode='w') as store:
            store['second_level'] = second_level
            store['first_level'] = first_level
            store['first_level_diagnostics'] = first_level_diagnostics
            if response_aligned:
                store['trial_counts'] = trial_counts
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
        store['first_level_diagnostics'] = first_level_diagnostics
        if response_aligned:
            store['trial_counts'] = trial_counts
except:
    warn('Results not saved yet!')
else:
    os.remove(tmpfile)
    
