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
import os


#%% options
# which dot to investigate?
dots = np.arange(1, 6)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = 0.1

# names of regressors needed to exclude bad trials
r_names = ['accev', 'accsur_pca', 'RT']
R = len(r_names)

# do not look for 100 ms spaced equal effects for trial regressors, instead 
# select a dot which defines the time points considered for the trial 
# regressors, e.g., if you select trialregs_dot=5 the regressors will be set to
# 0 for all data points not associated with the 5th dot in the selected 
# sequence, put differently: the trial regressors will only be applied to times
# get_data_times(t0) associated with the chosen dot which are t0+400 for the 
# 5th dot
# if trialregs_dot = 0, all the data from the dot sequence is used and you 
# essentially look for the average effect across the considered dot sequence
trialregs_dot = dots[-1]

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
file = os.path.join(helpers.resultsdir, 
        pd.datetime.now().strftime('evoked_meg_sequential'+'_%Y%m%d%H%M'+'.h5'))

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='zlib') as store:
    store['scalar_params'] = pd.Series([rt_thresh, trialregs_dot, sfreq, hfreq], 
         ['rt_thresh', 'trialregs_dot', 'sfreq', 'hfreq'])
    store['dots'] = pd.Series(dots)
    store['bl'] = pd.Series(bl)
    store['window'] = pd.Series(window)


#%% load all data
epochs_all = helpers.load_meg_epochs(hfreq, sfreq, window, bl=bl)
subjects = epochs_all.index.levels[0]
S = subjects.size

times = epochs_all.index.levels[2]

def get_data_times(t0):
    """Returns the times relative to a t0 when an effect of the chosen dots 
        can be expected. t0 is in ms!
    """
    datat = t0 + (dots - 1) * helpers.dotdt * 1000
    
    assert np.all(datat <= times[-1]), ("The times where you are looking for an "
                  "effect have to be within the specified time window")
             
    return times[[np.abs(times - t).argmin() for t in datat]]


#%% load trial-level design matrices for all subjects (ith-dot based)
DM = subject_DM.get_trial_DM(dots, r_names=r_names)
DM = DM.loc(axis=0)[subjects, :]

# remove trials which had too few simulations to estimate surprise
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

# create new good_trials index which repeats good_trials for the chosen number
# of dots, needed to select data below
good_trials = pd.concat([good_trials for d in dots], keys=dots, 
                        names=['dot'] + good_trials.index.names)
good_trials = good_trials.reorder_levels(['subject', 'trial', 'dot']).sort_index()

# don't need design matrix anymore
del DM


#%% normalise data
# this will include bad trials, too, but it shouldn't matter as this 
# normalisation across subjects, trials and time points is anyway very rough
# do this in two steps to save some memory
epochs_all = epochs_all - epochs_all.mean()
epochs_all = epochs_all / epochs_all.std()


#%% prepare output and helpers
tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)

erf = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              epochs_all.columns, times[:tendi+1]], 
              names=['permnr', 'channel', 'time']),
        columns=pd.Index(subjects, name='subject'))


#%% infer
for perm in np.arange(nperm+1):
    print('permutation %d' % perm)
    if perm > 0:
        # randomly permute time points, use different permutations for each trial, but use
        # the same permutation across all subjects such that only time points are permuted,
        # but random associations between subjects are maintained
        permutation = np.arange(epochs_all.shape[0]).reshape([x.size for x in epochs_all.index.levels])
        for tr in range(epochs_all.index.levels[1].size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, tr, :].T)
        epochs_all.loc[:] = epochs_all.values[permutation.flatten(), :]
        
    for t0 in times[:tendi+1]:
        print('t0 = %d' % t0)
        
        datat = get_data_times(t0)
        
        data = epochs_all.loc[(slice(None), slice(None), datat), :]
        data = data.loc[good_trials.values]
        
        erf.loc[(perm, slice(None), t0), :] = data.mean(level='subject').T.values
        
    try:
        with pd.HDFStore(file+'.tmp', mode='w') as store:
            store['erf'] = erf
    except:
        pass
    
    
if os.path.isfile(file+'.tmp'):
    os.remove(file+'.tmp')

with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
    store['erf'] = erf