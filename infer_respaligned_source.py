#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:30:26 2017

@author: bitzer
"""

import numpy as np
import helpers
import regressors
import subject_DM
import source_statistics as ss
import pandas as pd
import os
from scipy.stats import ttest_1samp
import statsmodels.api as sm
from warnings import warn


#%% options

# label mode = mean
srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201712061743.h5'

srcfile = os.path.join(ss.bem_dir, srcfile)

area = '4'

# set to 'local' to normalise within time points and subjects
# set to 'global' to normalise across the complete design matrix
# set to anything else to not normalise
# applies to both data and design matrix
normalise = 'local'

labels = ['L_%s_ROI-lh' % area, 'R_%s_ROI-rh' % area]

delays = np.arange(300, 510, 10, int)

resp_align = True

# allows you to exclude trials based on response-aligned time in first dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = 5000

if resp_align:
    timeslice = [-1000, 500]
    fbase = 'response-aligned'
else:
    timeslice = [0, 1500]
    fbase = 'firstdot-aligned'

times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                 name='time')

r_names = ['percupt_y_time', 'percupt_x_time', 'dot_x_time', 'dot_y_time',
           'intercept']
# sort regressor names so that the list has standard order used in DM later
r_names = sorted(r_names)

# where to store
file = pd.datetime.now().strftime('source_' + fbase + '_%Y%m%d%H%M' + '.h5')
# tmp-file should use local directory (preventing issue with remote store)
tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

# create HDF5-file and save chosen options
with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [toolate], 
            ['toolate'])
    store['normalise'] = pd.Series(normalise)
    store['srcfile'] = pd.Series(srcfile)


#%% load data time courses
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

choices = regressors.subject_trial.response.loc[subjects]
rts = regressors.subject_trial.RT.loc[subjects] * 1000

alldata, epochtimes = helpers.load_area_tcs(
        srcfile, subjects, labels, resp_align, rts, normalise)


#%%
first_level = pd.DataFrame([],
        index=pd.MultiIndex.from_product(
                [delays, labels, times], names=['delay', 'label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [subjects, ['beta', 'bse'], r_names], 
                names=['subject', 'measure', 'regressor']), dtype=float)
first_level.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [delays, labels, times], names=['delay', 'label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [['mean', 'std', 'tval', 'mlog10p'], r_names], 
                names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

data_counts = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [delays, times], names=['delay', 'time']),
        columns=pd.Index(subjects, name='subject'), dtype=float)


#%% standard summary statistics inference
for delay in delays:
    print('\ndelay = %d ms' % delay, flush=True)
    
    #%% create response-aligned design matrix
    DM = subject_DM.get_trial_time_DM(r_names, epochtimes.values / 1000, 
                                      delay=delay / 1000, subjects=subjects)
    
    DM.sort_index(axis=1, inplace=True)
    
    if normalise == 'global':
        DM = subject_DM.normalise_DM(DM, True)
    
    if resp_align:
        # exclude trials in which there was no response (timed out trials)
        def get_response_trials(sub):
            DMsub = DM.loc[sub]
            rtssub = rts.loc[sub]
            return DMsub.loc[
                    (rtssub[rtssub != helpers.toresponse[1] * 1000].index, 
                    slice(None)), :]
        
        DM = pd.concat([get_response_trials(sub) for sub in subjects],
                        keys=subjects, names=['subject', 'trial', 'time'])
        
    # now data and regressor are in the same order, so simply assign the 
    # response-aligned index of alldata to the regressor values to get 
    # response-aligned regressor values, if not response-aligned this transforms
    # the time index from seconds to milliseconds
    DM.index = alldata.index
    
    
    #%% run inference over all selected labels and time points
    for time in times:
        print('\rtime = % 5d' % time, end='', flush=True)
        
        trial_counts = []
        
        model = pd.concat([DM.xs(time, level='time'), 
                          alldata.xs(time, level='time')], axis=1)
        
        # remove trials which are > toolate to response
        if not resp_align:
            model = model.loc[rts[(time - rts) < toolate].index]
            
        model = model.dropna()
        
        # count remaining trials
        data_counts.loc[(delay, time), :] = model[r_names[0]].groupby(
                'subject').count()
    
        for label in labels:
            for sub in subjects:
                modelsub = model.loc[sub].copy()
                
                if modelsub.shape[0] > 30:
                    if normalise == 'local':
                        # subtracts the mean when its magnitude is more than 
                        # twice the std (see msratio)
                        modelsub[r_names] = subject_DM.normalise_DM(
                                modelsub[r_names], True)
                        
                        # only normalise std (mean to be capture by intercept)
                        modelsub[label] /= modelsub[label].std()
                    
                    res = sm.OLS(
                            modelsub[label].values, 
                            modelsub[first_level.columns.levels[2]].values, 
                            hasconst=True).fit()
                    first_level.loc[(delay, label, time), 
                                    (sub, 'beta', slice(None))] = res.params
                    first_level.loc[(delay, label, time),
                                    (sub, 'bse', slice(None))] = res.bse
            
            params = first_level.loc[(delay, label, time), 
                                     (slice(None), 'beta', slice(None))]
            params = params.unstack('regressor')
            
            params.dropna(inplace=True)
            
            second_level.loc[(delay, label, time), ('mean', slice(None))] = (
                    params.mean().values)
            second_level.loc[(delay, label, time), ('std', slice(None))] = (
                    params.std().values)
            tvals, pvals = ttest_1samp(params.values, 0, axis=0)
            second_level.loc[(delay, label, time), 
                             ('tval', slice(None))] = tvals
            second_level.loc[(delay, label, time), 
                             ('mlog10p', slice(None))] = -np.log10(pvals)
    
    try:
        with pd.HDFStore(tmpfile, mode='w') as store:
            store['second_level'] = second_level
            store['first_level'] = first_level
            store['data_counts'] = data_counts
    except:
        pass
        
try:
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['second_level'] = second_level
        store['first_level'] = first_level
        store['data_counts'] = data_counts
except:
    warn('Results not saved yet!')
else:
    os.remove(tmpfile)


#%% create Stan model
#hierarchical_stan = """
#data {
#    int<lower=1> S;
#    int<lower=0> N[S];
#    int<lower=1> Nall;
#     
#    vector[Nall] reg;
#    vector[Nall] y;
#}
#
#parameters {
#    real z;
#    real<lower=0> stdx;
#    
#    vector[S] x;
#    vector[S] intercept;
#    
#    vector<lower=0>[S] stdy;
#}
#
#transformed parameters {
#    vector[S] diffs = x * stdx;
#    vector[S] meany = diffs + z;
#}
#
#model {
#    int n = 0;
#    
#    z ~ normal(0, 1);
#    x ~ normal(0, 1);
#    intercept ~ normal(0, 1);
#    stdx ~ exponential(1);
#    stdy ~ exponential(1);
#    
#    for (sub in 1:S) {
#        y[n+1 : n+N[sub]] ~ normal(reg[n+1 : n+N[sub]] * meany[sub]
#                                   + intercept[sub], stdy[sub]);
#        
#        n = n + N[sub];
#    }
#}
#"""
#
#def get_stan_data(time, label):
#    data = pd.concat([sumx_normalised.xs(time, level='time'), 
#                      alldata.xs(time, level='time')[label]],
#             axis=1)
#    data = data.dropna()
#    N = data.sum_dot_x.groupby('subject').count()
#    
#    return {'S': N.size,
#            'N': N.values,
#            'Nall': N.sum(),
#            'reg': data.sum_dot_x.values,
#            'y': data[label].values}
#
## pre-compile
#fit = pystan.stan(model_code=hierarchical_stan, 
#                  data=get_stan_data(time, label),
#                  iter=1000, chains=4)
#
#
##%%
#fit2 = pystan.stan(fit=fit, data=get_stan_data(time, label),
#                   iter=1000, chains=4)