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
# allows you to exclude trials based on response-aligned time in dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = -200

# exclude timed-out trials
exclude_to = True

# only include subjects in second-level summary statistics analysis for who
# the number of trials available at the considered time point is at least:
mindata = 30

# names of regressors that should enter the GLM
r_names = ['percupt_x', 'percupt_y', 'abs_dot_y', 'abs_dot_x', 'dot_y', 
           'dot_x', 'intercept', 'accev', 'sum_dot_y_prev']
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

# dot onset aligned time window for which to run the analysis, 
# must be a 2-element list containing slice limits (right edge is inclusive)
# the right edge determines the maximum number of dots entering the analysis
# for each dot onset aligned time point by the difference to the time points
# available in the epochs, e.g., for epochs with 2490 ms as last time point and
# 700 ms as right edge of the dot onset aligned time window, dots 1-25 could
# enter the analysis
# note that normalisation of data is still over all times in srcfile
#timeslice = [0, 690]
timeslice = [0, 690]

# How many permutations should be computed?
nperm = 3

# set to 'local' for normalisation of DM within each t0, i.e., locally for each
#   regression that is run - means that each beta can be interpreted as the 
#   correlation value describing how strongly the regressor is represented in
#   the data at that time point and allows more or less fair comparison of 
#   regressor betas within and across each time point
# set to 'subject' to normalise the DM within each subject before entries are
#   excluded based on RTs or non-availability of data - means that betas 
#   reflect how strongly the regressor is represented in the data in relation
#   to the overall scale of the regressor; for regressors with theoretically
#   constant scale across dots (dots as independent samples from the same
#   distribution) this should lead to more accurate estimation of regressor
#   scale for normalisation, but if the scale of regressors is expected to
#   change across dots resulting betas may not be comparable; for example,
#   the scale of accumulated evidence rises with dot index meaning that at low
#   t0 regressor values will include more high values than at high t0s (because
#   for high t0s we exclude late dots from the analysis as they fall out of the
#   epoch time window, or are too late in relation to RT); if the true relation
#   between the regressor and the signal is linear and constant across time 
#   points, the estimated betas should be comparable, but if, for example, 
#   the relation to accev follows more a sigmoid where large values are mapped
#   to similar, moderately high signal values, the betas for early t0s may be
#   lower than the ones for late t0s, although it doesn't mean that accev is
#   less represented in the signal early after dot onset
normDM = 'local'

# whether to normalise the source signal to have mean 0 and std 1 in each area
# across trials, time points and subjects
normsrc = True

# source data to use

# label mode = mean, long epochs, HCPMMP_5_8
srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201712061743.h5'

# label mode = mean
#srcfile = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'

# label mode = mean_flip
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'

# label mode= None, lselection=[V1, V2], window=[0, 0.4]
#srcfile = 'source_epochs_allsubs_HCPMMP1_201708151431.h5'

# label mode = (abs) max
#srcfile = 'source_epochs_allsubs_HCPMMP1_201706271717.h5'

# label mode= None, lselection=pre-motor and motor areas, window=[0.3, 0.9]
#srcfile = 'source_epochs_allsubs_HCPMMP1_201710231731.h5'

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)

# where to store
file = pd.datetime.now().strftime('source_sequential'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [toolate, exclude_to, mindata, trialregs_dot, normsrc], 
            ['toolate', 'exclude_to', 'mindata', 'trialregs_dot', 'normsrc'])
    store['normDM'] = pd.Series(normDM)
    store['srcfile'] = pd.Series(srcfile)


#%% extract some basic info from example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

# times stored in epochs
epochtimes = epochs.index.levels[2]
# desired times for the analysis
times = epochtimes[slice(*epochtimes.slice_locs(*timeslice))]

# trials available in data (should be the same as in DM)
trials = epochs.index.get_level_values('trial').unique()

# figure out the maximum number of dots that we could consider in the analysis
# based on the available epoch times (absolute maximum is 25 by exp design)
dots = np.arange(1, min(25 + 1, epochtimes[-1] // (helpers.dotdt * 1000) + 2), 
                 dtype=int)

if normsrc:
    # get mean and std of data across all subjects, trials and times
    epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
    epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% generate dot-sequential DM including all possible dots
DM = subject_DM.get_trial_DM(dots, r_names=r_names)
DM = DM.loc[subjects, :]

def get_dot_DM(DM, dot):
    rn = list(map(lambda s: s+'_%d' % dot, r_cats['dotreg']))
    DMdot = DM[r_cats['trialreg'] + r_cats['constreg'] + rn]
    return DMdot.rename(columns=dict(zip(rn, r_cats['dotreg'])))

DM = pd.concat([get_dot_DM(DM, dot) for dot in dots], keys=dots, 
                names=['dot'] + DM.index.names)
DM = DM.reorder_levels(['subject', 'trial', 'dot']).sort_index()
DM.sort_index(axis=1, inplace=True)

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

# normalise within subject, if desired; 
# also centers the regressors, but only if their mean exceeds the standard 
# deviation by a given factor (2 by default)
if normDM == 'subject':
    DM = subject_DM.normalise_DM(DM, 'subject')

# only now set trial regressors to 0 for all except the chosen dot, because
# otherwise the calculation of the mean during normalisation would have been
# obscured
if trialregs_dot:
    if trialregs_dot < 0:
        trialregs_dot = dots[trialregs_dot]
    DM.loc[(slice(None), slice(None), np.setdiff1d(dots, trialregs_dot)), 
           r_cats['trialreg']] = 0


#%% create an index for subject-level design matrix for each of the possible
#   t0s in which the dot indeces are replaced with times (valid for all subs)
def get_DM_time_index(t0):
    """Returns an index for a subject-level DM with dot indeces replaced by 
       the corresponding first dot onset aligned times as stored in epochs.
       Times exceeding the time window available in epochs will be mapped to
       the last available time + 100
    """
    datat = t0 + (dots - 1) * helpers.dotdt * 1000
    
    # indeces of times that are in the time window stored in the epochs
    # account for numerical inaccuracies in the representation of times in
    # datat by adding a small amount to the last available time in epochs
    goodind = datat <= epochtimes[-1] + 1e-6
    
    # get the times associated with the dots, for dots that would lead to
    # times exceeding those available in epochs return last available time+100
    datatimes = np.r_[
            epochtimes[[np.abs(epochtimes - t).argmin() 
                        for t in datat[goodind]]],
            np.full((goodind==False).sum(), epochtimes[-1] + 100)]
    
    return pd.MultiIndex.from_arrays(
            [np.tile(trials, (dots.size, 1)).flatten(order='F'),
             np.tile(datatimes, trials.size)], 
            names=['trial', 'time'])
    

#%% load rts and choices of subjects
rts = subject_DM.regressors.subject_trial.RT.loc[subjects] * 1000
choices = subject_DM.regressors.subject_trial.response.loc[subjects]


#%% prepare time-dependent regressors
## index of last time point available for this analysis
#tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)
#
## include placeholders for time-dependent regressors in design matrix
#for r_name in r_cats['timereg']:
#    DM.insert(r_names.index(r_name), r_name, np.nan)
#
## load time-dependent regressors for each time point of the analysis
#def get_timereg(r_name):
#    regs = []
#    for t0 in times[:tendi+1]:
#        reg = subject_DM.regressors.subject_trial_time[r_name](get_data_times(t0) / 1000)
#        reg = reg.loc[list(subjects)].loc[good_trials.values]
#        regs.append(reg)
#        
#    regs = pd.concat(regs, axis=0, keys=times[:tendi+1], names=['t0']+regs[0].index.names)
#    
#    # I run the normalisation across all time points, because I want the 
#    # regressor to have the same scale across time points. Although the 
#    # regressor vector is much longer here than for the DM above, the scale 
#    # will be comparable to the other regressors, because the normalisation 
#    # corrects for the length of the vector so that individual elements have a
#    # scale close to 1
#    return subject_DM.normalise_DM(regs, 'subject')
#
#timeDM = {r_name: get_timereg(r_name) for r_name in r_cats['timereg']}
#
#def fill_in_timeDM(t0):
#    for r_name in timeDM.keys():
#        DM[r_name] = timeDM[r_name].loc[t0].values
#        
#    return DM


#%% prepare output and helpers
srclabels = epochs.columns

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

data_counts = pd.DataFrame(
        [], index=times, dtype=float,
        columns=pd.Index(subjects, name='subject'))

DM_condition_numbers = data_counts.copy()


#%% infer
# original sequential order
permutation = np.arange(480 * dots.size).reshape([480, dots.size])

for perm in np.arange(nperm+1):
    print('\npermutation %d' % perm)
    print('-------------')
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        for d in range(dots.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, d])
    
    for s, sub in enumerate(subjects):
        with pd.HDFStore(srcfile, 'r') as store:
            epochs = store.select('label_tc', 'subject=sub').loc[sub]
            
        # normalise each label across time points and subjects
        if normsrc:
            epochs = (epochs - epochs_mean) / epochs_std
        
        for t0 in times:
            print('\rsubject = %2d, t0 = %3d' % (sub, t0), end='', flush=True)
            
            DMsub = DM.loc[sub].copy()
            DMsub.index = get_DM_time_index(t0)
            
            model = pd.concat([epochs.loc[DMsub.index], DMsub], axis=1,
                               keys=['data', 'DM'], names=['var', 'col'])
            
            # NaN all values for which time points are too late;
            # only need to do that for one column, because rows are later
            # excluded based on whether there is a nan in any column
            for tr in trials:
                # small offset added to ensure that only times > RT removed
                model.loc[(tr, slice(rts.loc[(sub, tr)] + toolate + 1e-6, 
                                     None)), model.columns[0]] = np.nan
            
            # NaN all values for trials that were timed out, if desired
            if exclude_to:
                ch_sub = choices.loc[sub]
                model.loc[ch_sub[ch_sub == 0].index.values, 
                          model.columns[0]] = np.nan
            
#            for r_name in timeDM.keys():
#                DM.loc[sub, r_name] = timeDM[r_name].loc[t0].loc[sub].values
            
            # permute trials in the data
            model['data'] = model['data'].values[permutation.flatten(), :]
            
            # remove the excluded rows
            model.dropna(inplace=True)
            
            # normalise
            if normDM == 'local':
                model.DM = subject_DM.normalise_DM(model.DM, True)
            
            # save data count and condition number
            if perm == 0:
                data_counts.loc[t0, sub] = model.shape[0]
                
                # numbers from 10 indicate some problem with collinearity in 
                # design matrix (see Rawlings1998, p. 371)
                # when motoprep is included, high condition numbers at early 
                # time points (<100 ms) result from motoprep only having very 
                # few non-zero values due to late responses of some subjects
                DM_condition_numbers.loc[t0, sub] = (
                        subject_DM.compute_condition_number(model.DM))
                
                if DM_condition_numbers.loc[t0, sub] > 10:
                    warn("High condition number of design matrix detected "
                         "(%5.2f) for subject %d, t0 = %d!" % (
                                 DM_condition_numbers.loc[t0, sub], sub, t0))
            
            # run regression for each source label, but only for data sets
            # with at least as many data points as there are regressors
            if model.shape[0] >= model.DM.shape[1]:
                for label in srclabels:
                    res = sm.OLS(model[('data', label)].values, 
                                 model.DM.values, hasconst=True).fit()
                
                    first_level.loc[(perm, label, t0), 
                                    (sub, 'beta', slice(None))] = res.params
                    first_level.loc[(perm, label, t0), 
                                    (sub, 'bse', slice(None))] = res.bse
                    
                    first_level_diagnostics.loc[(perm, label, t0), 
                                                (sub, 'Fval')] = res.fvalue
                    first_level_diagnostics.loc[(perm, label, t0), 
                                                (sub, 'R2')] = res.rsquared
                    first_level_diagnostics.loc[(perm, label, t0), 
                                                (sub, 'llf')] = res.llf
    
    print('\ncomputing second level ...')
    for label in srclabels:
        for t0 in times:
            params = first_level.loc[(perm, label, t0), 
                                     (slice(None), 'beta', slice(None))]
            params = params.unstack('regressor')
            
            # exclude subjects with less than mintrials trials at t0
            params = params[(data_counts.loc[t0] >= mindata).values]
            
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
            store['data_counts'] = data_counts
            store['DM_condition_numbers'] = DM_condition_numbers
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
        store['first_level_diagnostics'] = first_level_diagnostics
        store['data_counts'] = data_counts
        store['DM_condition_numbers'] = DM_condition_numbers
except:
    warn('Results not saved yet!')
else:
    os.remove(tmpfile)
    
