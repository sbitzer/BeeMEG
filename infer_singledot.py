#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:38:01 2018

@author: bitzer
"""

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
import os, gc, re
import statsmodels.api as sm
from warnings import warn


#%% options
# which dots to investigate? Note that the more dots you include, the fewer
# trials will be available in the regression, if you exclude trials based on
# toolate below
dots = np.arange(1, 7)

# allows you to exclude trials based on response-aligned time in dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = 5000

# exclude timed-out trials
exclude_to = True

# only include subjects in second-level summary statistics analysis for who
# the number of trials available at the considered time point is at least:
mindata = 30

# names of regressors that should enter the GLM
r_names = ['trial_time', 'intercept', 'response']
R = len(r_names)
# sort for use in index below
r_names.sort()
# get associated regressor categories
r_cats = subject_DM.regressors.get_regressor_categories(r_names)

# whether a response-aligned analysis should be run, i.e., whether time should
# be defined with respect to the response time; if True, set timeslice below
# providing both ends
response_aligned = True
if response_aligned and toolate <= 1000:
    key = input("toolate is suspiciously low ({}) for response-aligned "
                "analysis!\nContinue anyway? (y/n): ".format(toolate))
    if key.lower() != "y":
        raise ValueError("toolate too low !")

# time window for which to run the analysis, must be a 0-, 1- or 2-element list
# if empty list, all times in srcfile are used for analysis;
# if 1-element, all times starting with the given will be included in analysis;
# if 2-element, these are the slice limits considered (right edge is inclusive)
# note that global normalisation of data is still over all times in srcfile
timeslice = [-1000, 500]

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

# set to 'global' to normalise the data to have mean 0 and std 1 in 
#   each data label across trials, time points and subjects
# set to 'trials' to normalise to mean 0, std 1 within each subject and time
#   point, i.e., normalise only across trials; together with normDM=local 
#   allows to interpret estimated coefficients as correlations
normdata = 'trials'

# data to use, 'meg' for MEG channels, 'source' for sources
datatype = 'meg'

if datatype == 'meg':
    sfreq = 100
    megtype = 'mag'
    eog = False
    bl = (-0.3, 0)
    window = [0, 2.5]
    
    srcfile = helpers.create_meg_epochs_hdf(sfreq, sfreq, window, megtype, bl,
                                            eog=eog)
    epname = 'epochs'
    
elif datatype == 'source':
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
    
    # label mode = None, lselection=premotor, motor, mid cingulate, parietal
    # window = [-0.3, 2.5], no baseline correction before source reconstruction
#    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201802081827.h5'
    
    srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)
    epname = 'label_tc'

# where to store
file = pd.datetime.now().strftime(datatype+'_singledot'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [toolate, exclude_to, mindata, float(response_aligned)], 
            ['toolate', 'exclude_to', 'mindata', 'response_aligned'])
    store['dots'] = pd.Series(dots)
    store['normDM'] = pd.Series(normDM)
    store['normdata'] = pd.Series(normdata)
    store['srcfile'] = pd.Series(srcfile)


#%% extract some basic info from example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select(epname, 'subject=2')

# times stored in epochs
epochtimes = epochs.index.levels[2]

# trials available in data (should be the same as in DM)
trials = epochs.index.get_level_values('trial').unique()

if normdata == 'global':
    # get mean and std of data across all subjects, trials and times
    epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
    epochs_std = pd.read_hdf(srcfile, 'epochs_std')

gc.collect()


#%% load rts and choices of subjects
rts = subject_DM.regressors.subject_trial.RT.loc[subjects] * 1000
choices = subject_DM.regressors.subject_trial.response.loc[subjects]


#%% figure out which time points to analyse
if response_aligned:
    if len(timeslice) != 2:
        raise ValueError("For response aligned times you need to provide the "
                         "start and stop of the time window!")
    times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                     name='time')
else:
    times = epochtimes[slice(*epochtimes.slice_locs(*timeslice))]

# default index for epochs, used for reindexing below to select desired times
index = pd.MultiIndex.from_product([trials, times], names=['trial', 'time'])


#%% load trial-level design matrices for all subjects (ith-dot based)
DM = subject_DM.get_trial_DM(dots, subjects, r_names=r_names)

# add selected single dot regressors (if you specifically requested, e.g., 
# dot_x_9 in r_names)
single_r_names = []
single_r_names_components = []
for r_name in r_names:
    match = re.match(r'(.*)_(\d)$', r_name)
    if match:
        single_r_names.append(r_name)
        single_r_names_components.append(match.groups())

DM = pd.concat(
        [DM]+[subject_DM.get_trial_DM(int(dot), subjects, r_names=[r_name]) 
              for r_name, dot in single_r_names_components], axis=1)

# move_dist_1 is constant at 0, we don't need to fit that
if 'move_dist' in r_names:
    del DM['move_dist_1']
    
# sort regressor names (columns) for indexing
DM.sort_index(axis=1, inplace=True)

# normalise within subject, if desired;
# also centers the regressors, but only if their mean exceeds the standard 
# deviation by a given factor (2 by default)
if normDM == 'subject':
    DM = subject_DM.normalise_DM(DM, 'subject')

if DM.isnull().any().any():
    raise ValueError("Design matrix has NaNs somewhere, please check!")


#%% prepare output and helpers
datalabels = epochs.columns

first_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              datalabels, times], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['beta', 'bse'], 
              DM.columns], names=['subject', 'measure', 'regressor']), 
              dtype=np.float64)
first_level.sort_index(axis=1, inplace=True)

first_level_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              datalabels, times], 
              names=['permnr', 'label', 'time']),
        columns=pd.MultiIndex.from_product([subjects, ['Fval', 'R2', 'llf']], 
              names=['subject', 'measure']), dtype=np.float64)
first_level_diagnostics.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product([np.arange(nperm+1), 
              datalabels, times], 
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
permutation = np.arange(trials.size * times.size).reshape(
        [trials.size, times.size])

for perm in range(0, nperm+1):
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
        with pd.HDFStore(srcfile, 'r') as store:
            epochs = store.select(epname, 'subject=sub')
        gc.collect()
        
        # normalise each label, if desired
        if normdata == 'global':
            epochs = (epochs - epochs_mean) / epochs_std
        elif normdata == 'trials':
            epochs = epochs.unstack('time')
            epochs = (epochs - epochs.mean()) / epochs.std()
            epochs = epochs.stack('time')
        
        # NaN all values for trials that were timed out, if desired;
        # only necessary for one column as later rows are excluded based on
        # nan in any column
        if exclude_to:
            ch_sub = choices.loc[sub]
            toind = ch_sub[ch_sub == helpers.toresponse[0]].index.values
            if toind.size > 0:
                epochs.loc[(sub, toind), epochs.columns[0]] = np.nan
            
        # NaN within each trial time points that are "too late" with respect to
        # response aligned time
        for tr in trials:
                # small offset added to ensure that only times > RT removed
                epochs.loc[
                        (sub, tr, slice(rts.loc[(sub, tr)] + toolate + 1e-6, None)), 
                        epochs.columns[0]] = np.nan
                        
        # change to response-aligned times, when desired
        if response_aligned:
            epochs.index = helpers.to_resp_aligned_index(epochs.index, rts)
        
        # from here on I don't need the subject index
        epochs = epochs.loc[sub]
            
        # select desired times
        if len(timeslice) > 0:
            epochs = epochs.reindex(index)
            
        # permute by reordering rows according to sampled permutation
        epochs.loc[:] = epochs.values[permutation.flatten(), :]
        
        # drop nans to implement exclusion of undesired data points
        epochs.dropna(inplace=True)
        
        # save data counts for this subject
        if perm == 0:
            data_counts[sub] = epochs.iloc[:, 0].groupby('time').count()
        
        for t0 in times:
            print('\rsubject = %2d, t0 = % 5d' % (sub, t0), end='', flush=True)
            
            data = epochs.xs(t0, level='time')
            DMt0 = DM.loc[sub].loc[data.index]
            
            # normalise DM only across given trials, if desired
            if normDM == 'local':
                DMt0 = subject_DM.normalise_DM(DMt0, True)
            
            # save data count and condition number
            if perm == 0:
                # numbers from 10 indicate some problem with 
                # collinearity in design matrix (see Rawlings1998, p. 371)
                # when motoprep is included, high condition numbers at early 
                # time points (<100 ms) result from motoprep only having very 
                # few non-zero values due to late responses of some subjects
                DM_condition_numbers.loc[t0, sub] = (
                        subject_DM.compute_condition_number(DMt0))
                
                if DM_condition_numbers.loc[t0, sub] > 10:
                    warn("High condition number of design matrix detected "
                         "({:5.2f}) for subject {:d}, t0 = {:d}!".format(
                                 DM_condition_numbers.loc[t0, sub], sub, t0))
            
            # run regression for each data label, but only for data sets
            # with at least as many data points as there are regressors
            if data.shape[0] >= DMt0.shape[1]:
                for label in datalabels:
                    res = sm.OLS(data[label].values, 
                                 DMt0.values, hasconst=True).fit()
                
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
    for label in datalabels:
        for t0 in times:
            params = first_level.loc[(perm, label, t0), 
                                     (slice(None), 'beta', slice(None))]
            params = params.unstack('regressor')
            
            # exclude subjects with less than mintrials trials at t0 or 
            # DM condition numbers greater 10
            params = params[(data_counts.loc[t0] >= mindata).values &
                            (DM_condition_numbers.loc[t0] <= 10).values]
            
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
    
