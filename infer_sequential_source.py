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
dots = np.arange(1, 4)

# implements the assumption that in a trial with RT<(dot onset time + rt_thresh)
# there cannot be an effect of the last considered dot in the MEG signal
rt_thresh = 0.1

# names of regressors that should enter the GLM
r_names = ['abs_dot_y', 'abs_dot_x', 'dot_y', 'dot_x', 'entropy', 'trial_time', 
           'intercept', 'accev', 'accsur_pca', 'response', 'dot_x_cflip', 
           'accev_cflip', 'move_dist', 'sum_dot_y_prev']
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
# if trialregs_dot = -1 you use the last considered dot
trialregs_dot = -1

# source data to use
# label mode = mean_flip
srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'
# label mode = (abs) max
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706271717.h5'

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)

# How many permutations should be computed?
nperm = 3

# whether to normalise the source signal to have mean 0 and std 1 in each area
# across trials, time points and subjects
normsrc = True

# where to store
file = pd.datetime.now().strftime('source_sequential'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series([rt_thresh, trialregs_dot, normsrc], 
         ['rt_thresh', 'trialregs_dot', 'normsrc'])
    store['dots'] = pd.Series(dots)
    store['srcfile'] = pd.Series(srcfile)


#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

times = epochs.index.levels[2]

def get_data_times(t0):
    """Returns the times relative to a t0 when an effect of the chosen dots 
        can be expected. t0 is in ms!
    """
    datat = t0 + (dots - 1) * helpers.dotdt * 1000
    
    assert np.all(datat <= times[-1]), ("The times where you are looking for an "
                  "effect have to be within the specified time window")
             
    return times[[np.abs(times - t).argmin() for t in datat]]


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
trialregs = list(np.setdiff1d(r_names, dotregs))
        
def get_dot_DM(DM, dot):
    rn = list(map(lambda s: s+'_%d' % dot, dotregs))
    DMdot = DM[trialregs + rn]
    return DMdot.rename(columns=dict(zip(rn, dotregs)))

DM = pd.concat([get_dot_DM(DM, dot) for dot in dots], keys=dots, 
                names=['dot'] + DM.index.names)
DM = DM.reorder_levels(['subject', 'trial', 'dot']).sort_index()
DM.sort_index(axis=1, inplace=True)


#%% normalise data and regressors to simplify interpretation
# find regressors that have constant value for the first dot
dmstds = DM.loc[(subjects[0], slice(None), 1)].std()
constregs = dmstds[np.isclose(dmstds, 0)].index
# don't care about intercept
constregs = constregs[constregs != 'intercept']

# only dot-level regressor should be constant for the first dot, 
# otherwise something's wrong
assert np.setdiff1d(constregs, dotregs).size == 0

otherregs = np.setdiff1d(r_names, list(constregs)+['intercept'])

# for constregs I need to exclude first dot data from normalisation
DM2const = DM.loc[(slice(None), slice(None), slice(2, dots[-1])), constregs]
DM[constregs] = (DM[constregs] - DM2const.mean()) / DM2const.std()
del DM2const

# then set first dot values of constregs to 0 to exclude that they have any
# influence on the estimation of the regressor betas
DM.loc[(slice(None), slice(None), 1), constregs] = 0

# other regressors can be normalised in the standard way
DM[otherregs] = (DM[otherregs] - DM[otherregs].mean()) / DM[otherregs].std()

# set trial regressors to 0 for all except the chosen dot
if trialregs_dot:
    if trialregs_dot < 0:
        trialregs_dot = dots[trialregs_dot]
    DM.loc[(slice(None), slice(None), np.setdiff1d(dots, trialregs_dot)), 
           list(np.setdiff1d(trialregs, 'intercept'))] = 0

# just get mean and std of data across all subjects, trials and times
epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% prepare output and helpers
tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)

srclabels = epochs_mean.index

first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times[:tendi+1]], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              DM.columns], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times[:tendi+1]], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              srclabels, times[:tendi+1]], 
              names=['permnr', 'label', 'time']), 
        columns=pd.MultiIndex.from_product([['mean', 'std', 'mlog10p'], 
              DM.columns], names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

assert np.all(first_level.columns.levels[2] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'
assert np.all(second_level.columns.levels[1] == DM.columns), 'order of ' \
    'regressors in design matrix should be equal to that of results DataFrame'


#%% infer
# create new good_trials index which repeats good_trials for the chosen number
# of dots, needed to select data below
good_trials = pd.concat([good_trials for d in dots], keys=dots, 
                        names=['dot'] + good_trials.index.names)
good_trials = good_trials.reorder_levels(['subject', 'trial', 'dot']).sort_index()

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
            
            # normalise each label across time points and subjects
            if normsrc:
                epochs = (epochs - epochs_mean) / epochs_std
            
            epochs.loc[:] = epochs.values[permutation.flatten(), :]
            
            for t0 in times[:tendi+1]:
                print('\rsubject = %2d, t0 = %3d' % (sub, t0), end='')
        
                datat = get_data_times(t0)
                
                data = epochs.loc[(sub, slice(None), datat)]
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
        for t0 in times[:tendi+1]:
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
    
