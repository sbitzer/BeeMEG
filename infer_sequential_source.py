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

# excludes data which occured at time points after RT - rt_thresh
rt_thresh = 0.5

# names of regressors that should enter the GLM
r_names = ['abs_dot_y', 'abs_dot_x', 'dot_y', 'dot_x', 'intercept', 
           'accev', 'sum_dot_y_prev']
R = len(r_names)
# sort for use in index below
r_names.sort()
# get associated regressor categories
r_cats = subject_DM.regressors.get_regressor_categories(r_names)

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
trialregs_dot = 0

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
DM = subject_DM.get_trial_DM(dots, r_names=r_names)
    
DM = DM.loc(axis=0)[subjects, :]

# remove trials which had too few simulations to estimate surprise
snames = ['accev_%d' % d for d in dots]
good_trials = np.logical_not(np.any(np.isnan(DM[snames]), axis=1))
if 'accsur_pca_%d' % dots[0] in DM.columns:
    snames = ['accsur_pca_%d' % d for d in dots]
    good_trials = np.logical_and(good_trials, 
            np.logical_not(np.any(np.isnan(DM[snames]), axis=1)))
good_trials = pd.Series(good_trials, index=DM.index)
                         
# select only the good trials
DM = DM.loc[good_trials]

# create new good_trials index which repeats good_trials for the chosen number
# of dots, needed to select data below
good_trials = pd.concat([good_trials for d in dots], keys=dots, 
                        names=['dot'] + good_trials.index.names)
good_trials = good_trials.reorder_levels(['subject', 'trial', 'dot']).sort_index()


#%% get RTs for later exclusion of data based on closeness to RT
rts = pd.concat(
        [subject_DM.regressors.subject_trial.RT.loc[subjects]] * dots.size,
        keys=dots, names=['dot', 'subject', 'trial'])
rts = rts.reorder_levels(['subject', 'trial', 'dot']).sort_index()

rts = rts[good_trials]


#%% reshape DM to have the dots in rows
def get_dot_DM(DM, dot):
    rn = list(map(lambda s: s+'_%d' % dot, r_cats['dotreg']))
    DMdot = DM[r_cats['trialreg'] + r_cats['constreg'] + rn]
    return DMdot.rename(columns=dict(zip(rn, r_cats['dotreg'])))

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
assert np.setdiff1d(constregs, r_cats['dotreg']).size == 0

# set first dot values of constregs to 0 to exclude that they have any
# influence on the estimation of the regressor betas, this is a precaution as
# it can be for single subjects that the constant value for the first dot has
# a noticable influence on the overall effect, but this influence is then 
# mainly due to the constant regressors picking up the average signal through
# the collinearity with the intercept
DM.loc[(slice(None), slice(None), 1), constregs] = 0

# now normalise the scale of all regressors within subjects, this also centers
# the regressors, but only if their mean exceeds the standard deviation by a 
# given factor (2 by default)
DM = subject_DM.normalise_DM(DM, 'subject')

# only now set trial regressors to 0 for all except the chosen dot, because
# otherwise the calculation of the mean during normalisation would have been
# obscured
if trialregs_dot:
    if trialregs_dot < 0:
        trialregs_dot = dots[trialregs_dot]
    DM.loc[(slice(None), slice(None), np.setdiff1d(dots, trialregs_dot)), 
           r_cats['trialreg']] = 0

# just get mean and std of data across all subjects, trials and times
epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% prepare time-dependent regressors
# index of last time point available for this analysis
tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)

# include placeholders for time-dependent regressors in design matrix
for r_name in r_cats['timereg']:
    DM.insert(r_names.index(r_name), r_name, np.nan)

# load time-dependent regressors for each time point of the analysis
def get_timereg(r_name):
    regs = []
    for t0 in times[:tendi+1]:
        reg = subject_DM.regressors.subject_trial_time[r_name](get_data_times(t0) / 1000)
        reg = reg.loc[list(subjects)].loc[good_trials.values]
        regs.append(reg)
        
    regs = pd.concat(regs, axis=0, keys=times[:tendi+1], names=['t0']+regs[0].index.names)
    
    # I run the normalisation across all time points, because I want the 
    # regressor to have the same scale across time points. Although the 
    # regressor vector is much longer here than for the DM above, the scale 
    # will be comparable to the other regressors, because the normalisation 
    # corrects for the length of the vector so that individual elements have a
    # scale close to 1
    return subject_DM.normalise_DM(regs, 'subject')

timeDM = {r_name: get_timereg(r_name) for r_name in r_cats['timereg']}


#%% check collinearity in design matrix using condition number
def fill_in_timeDM(t0):
    for r_name in timeDM.keys():
        DM[r_name] = timeDM[r_name].loc[t0].values
        
    return DM

# numbers from 10 indicate some problem with collinearity in design matrix
# (see Rawlings1998, p. 371)
# when motoprep is included, high condition numbers at early time points 
# (<100 ms) result from motoprep only having very few non-zero values due to
# late responses of some subjects
DM_condition_numbers = pd.concat(
        [subject_DM.compute_condition_number(
                fill_in_timeDM(t0), scope='subject') 
         for t0 in times[:tendi+1]],
        axis=1, keys=times[:tendi+1])

with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
    store['DM_condition_numbers'] = DM_condition_numbers

# warn, if there are more than 2 subjects with condition numbers above 10 at
# any time point from 100 ms within the trial
if np.any((DM_condition_numbers.loc[:, 100:] > 10).sum() > 2):
    warn("High condition numbers of design matrix detected!")


#%% prepare output and helpers
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
        columns=pd.MultiIndex.from_product([['mean', 'std', 'tval', 'mlog10p'], 
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
            
            # normalise each label across time points and subjects
            if normsrc:
                epochs = (epochs - epochs_mean) / epochs_std
            
            epochs.loc[:] = epochs.values[permutation.flatten(), :]
            
            for t0 in times[:tendi+1]:
                print('\rsubject = %2d, t0 = %3d' % (sub, t0), end='', flush=True)
        
                datat = get_data_times(t0)
                
                data = epochs.loc[(sub, slice(None), datat)]
                data = data.loc[good_trials.loc[sub].values]
                
                # exclude data points after RT-rt_thresh
                if rt_thresh is not None:
                    dottimes = np.tile(datat / 1000, 480)
                    dottimes = dottimes[good_trials.loc[sub].values]
                    select = ((rts.loc[sub] - dottimes) > 0.5).values
                    
                    data = data[select]
                
                for r_name in timeDM.keys():
                    DM.loc[sub, r_name] = timeDM[r_name].loc[t0].loc[sub].values
                
                if rt_thresh is None:
                    DMsubt0 = DM.loc[sub].values
                else:
                    DMsubt0 = DM.loc[sub].values[select, :]
        
                for label in srclabels:
                    res = sm.OLS(data[label].values, DMsubt0, 
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
    
