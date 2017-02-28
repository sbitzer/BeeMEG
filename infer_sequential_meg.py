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
# where to store
file = os.path.join(helpers.resultsdir, 'meg_sequential.h5')

# which dot to investigate?
dots = np.arange(1, 6)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = 0.1

# names of regressors that should enter the GLM
r_names = ['abs_dot_y', 'abs_dot_x', 'dot_y', 'dot_x', 'entropy', 'trial_time', 
           'intercept', 'accev', 'accsur_pca']
R = len(r_names)

# baseline period (set to None for no baseline correction)
bl = (-0.3, 0)

# desired sampling frequency of data
sfreq = 100

# smoothing frequency; the lower, the more smoothing; set to sfreq for no 
# smoothing, otherwise can be maximally sfreq/2
hfreq = sfreq

# time window within an epoch to consider
window = [0, 0.9]

# permute the data?
permute = False

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='zlib') as store:
    store['dots'] = pd.Series(dots)
    store['rt_thresh'] = pd.Series(rt_thresh)
         

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

# randomly permute trials, use different permutations for time points, but use
# the same permutation across all subjects such that only trials are permuted,
# but random associations between subjects are maintained
if permute:
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
# remove RT from design matrix
if 'RT' not in r_names:
    del DM['RT']

# select only the good trials
DM = DM.loc[good_trials]


#%% reshape DM to have the dots in rows
dotregs = []
for col in DM.columns:
    if col[-2:] == '_%d' % dots[0]:
        dotregs.append(col[:-2])
otherregs = list(np.setdiff1d(r_names, dotregs))
        
def get_dot_DM(DM, dot):
    rn = list(map(lambda s: s+'_%d' % dot, dotregs))
    DMdot = DM[otherregs + rn]
    return DMdot.rename(columns=dict(zip(rn, dotregs)))

DM = pd.concat([get_dot_DM(DM, dot) for dot in dots], keys=dots, 
                names=['dot'] + DM.index.names)
DM = DM.reorder_levels(['subject', 'trial', 'dot']).sort_index()
DM.sort_index(axis=1, inplace=True)


#%% normalise data and regressors to simplify definition of priors
DM = (DM - DM.mean()) / DM.std()
# intercept will be nan, because it has no variance
DM['intercept'] = 1

# this will include bad trials, too, but it shouldn't matter as this 
# normalisation across subjects, trials and time points is anyway very rough
epochs_all = (epochs_all - epochs_all.mean()) / epochs_all.std()


#%% prepare output and helpers
tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([epochs_all.columns, 
              times[:tendi+1]], names=['channel', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% infer
# create new good_trials index which repeats good_trials for the chosen number
# of dots, needed to select data below
good_trials = pd.concat([good_trials for d in dots], keys=dots, 
                        names=['dot'] + good_trials.index.names)
good_trials = good_trials.reorder_levels(['subject', 'trial', 'dot']).sort_index()

for t0 in times[:tendi+1]:
    print('t0 = %d' % t0)
    
    datat = get_data_times(t0)
    
    for channel in epochs_all.columns:
        data = epochs_all.loc[(slice(None), slice(None), datat), channel]
        data = data.loc[good_trials.values]
            
        params = np.zeros((S, R))
        for s, sub in enumerate(subjects):
            res = sm.OLS(data.loc[sub].values, DM.loc[sub].values, 
                         hasconst=True).fit()
            params[s, :] = res.params
           
        second_level.loc[(channel, t0), ('mean', slice(None))] = (
                params.mean(axis=0))
        second_level.loc[(channel, t0), ('std', slice(None))] = (
                params.std(axis=0))
        _, pvals = ttest_1samp(params, 0, axis=0)
        second_level.loc[(channel, t0), ('mlog10p', slice(None))] = (
                -np.log10(pvals))
        
    try:
        with pd.HDFStore(file, mode='r+', complevel=7, complib='zlib') as store:
            store['second_level'] = second_level
    except:
        pass