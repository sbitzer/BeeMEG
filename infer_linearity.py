#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:50:59 2017

@author: bitzer
"""

import numpy as np
import helpers
import regressors
import subject_DM
import source_statistics as ss
import pandas as pd
import os
import pystan
import psis
import warnings


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

delays = [330, 370, 400, 430, 470]

# whether to do response aligned analysis
resp_align = True

# whether to exclude trials that were timed out (resp_align=True overwrites
# this setting with exclude_to=True)
exclude_to = True

# allows you to exclude trials based on response-aligned time in first dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = 5000

if resp_align:
    timeslice = [-350, min(min(delays), -350)]
    fbase = 'response-aligned'
else:
    timeslice = [max(delays), 1500]
    fbase = 'firstdot-aligned'

times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                 name='time')

fbase = 'linearity_%s_%s_norm%s_toolate%+05d' % (
            fbase, area, normalise, toolate)
fstore = os.path.join(
        helpers.resultsdir, 
        pd.datetime.now().strftime('source_linearity'+'_%Y%m%d%H%M'+'.h5'))

r_names = ['dot_x_time', 'intercept']
r_name = set(r_names).difference({'intercept'}).pop()

use_loodiff_mean = True

# number of total samples per chain, the first half of samples will be excluded
# as warmup
nsamples = 1000

# whether to skip the mixed models (dotx_step and choice_linear)
skip_mix = True

# how many different chains to run, give a single number if Stan should run the
# chains in parallel itself, give a list or array, if chains should be run in
# sequence outside of Stan (to prevent memory problems)
nchains = np.ones(4)
nchains = np.atleast_1d(nchains).astype(int)

with pd.HDFStore(fstore, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [resp_align, toolate, nsamples, nchains, use_loodiff_mean, 
             exclude_to], 
            ['resp_align', 'toolate', 'nsamples', 'nchains',
             'use_loodiff_mean', 'exclude_to'])
    store['nchains'] = pd.Series(nchains)
    store['delays'] = pd.Series(delays)
    store['r_name'] = pd.Series(r_name)
    store['srcfile'] = pd.Series(srcfile)
    store['normalise'] = pd.Series(normalise)


#%% load data time courses
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

choices = regressors.subject_trial.response.loc[subjects]
rts = regressors.subject_trial.RT.loc[subjects] * 1000

alldata, epochtimes = helpers.load_area_tcs(
        srcfile, subjects, labels, resp_align, rts, normalise, exclude_to)


#%% load regressor time course
DM = pd.concat(
        [subject_DM.get_trial_time_DM(
                [r_name], epochtimes.values / 1000, delay=delay / 1000, 
                subjects=subjects, exclude_to=exclude_to, 
                resp_align=resp_align)
         for delay in delays], 
        keys=delays, names=['delay', 'subject', 'trial', 'time'])

DM.sort_index(axis=1, inplace=True)

if normalise == 'global':
    DM = subject_DM.normalise_DM(DM, True)

# remove timed out trials from rts, if they were also removed from the data and
# DM; this is necessary, because rts are later used to index trials in the 
# data that are too late
if exclude_to:
    rts = rts[rts != helpers.toresponse[1] * 1000]


#%% define and pre-compile the Stan model
simple = """
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real beta;
    real c;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu = beta * 2 * (inv_logit(x) - 0.5) + c;
}

model {
    beta ~ normal(0, 1);
    c ~ normal(0, 1);
    sigma ~ exponential(1);
    
    y ~ normal(mu, sigma);
}
"""

step_stan = """
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real beta;
    real<lower=0> l;
    real c;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu = beta * 2 * (inv_logit(x / l) - 0.5) + c;
}

model {
    beta ~ normal(0, 1);
    c ~ normal(0, 1);
    sigma ~ exponential(1);
    l ~ normal(0, 1);
    
    y ~ normal(mu, sigma);
}
"""

hierarchical_step_stan = """
data {
    int<lower=1> S;
    int<lower=0> N[S];
    int<lower=1> Nall;
    
    vector[Nall] x;
    vector[Nall] y;
}

parameters {
    real beta;
    real<lower=0> l;
    
    vector[S] beta_diff_norm;
    real<lower=0> std_beta;
    
    vector[S] intercept;
    vector<lower=0>[S] stdy;
}

transformed parameters {
    vector[S] beta_sub = beta + beta_diff_norm * std_beta;
}

model {
    int n = 0;
    vector[Nall] mu;
    vector[Nall] std;
       
    beta ~ normal(0, 1);
    l ~ normal(0, 1);
    
    beta_diff_norm ~ normal(0, 1);
    std_beta ~ exponential(1);
    
    intercept ~ normal(0, 1);
    stdy ~ exponential(1);
    
    for (sub in 1:S) {
        mu[n+1 : n+N[sub]] = beta_sub[sub] * 2 * (
                inv_logit(x[n+1 : n+N[sub]] / l) - 0.5)
                + intercept[sub];
        
        std[n+1 : n+N[sub]] = rep_vector(stdy[sub], N[sub]);
        
        n = n + N[sub];
    }
    
    y ~ normal(mu, std);
}
"""

linstepmix = """
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] signx;
    vector[N] y;
    
    real<lower=0> ab_beta;
}

parameters {
    real beta_lin;
    real beta_step;
    real<lower=0, upper=1> l;
    real c;
    real<lower=0> sigma;
}

model {
    vector[N] mu_lin = beta_lin * x + c;
    vector[N] mu_step = beta_step * signx + c;

    beta_lin ~ normal(0, 1);
    beta_step ~ normal(0, 1);
    c ~ normal(0, 1);
    sigma ~ exponential(1);
    l ~ beta(ab_beta, ab_beta);
    
    target += log_mix(l, 
                      normal_lpdf(y | mu_lin, sigma),
                      normal_lpdf(y | mu_step, sigma));
}
"""

hierarchical_linstepmix = """
data {
    int<lower=1> S;
    int<lower=0> N[S];
    int<lower=1> Nall;
    
    vector[Nall] x;
    vector[Nall] signx;
    vector[Nall] y;
    
    real<lower=0> ab_beta;
}

parameters {
    real beta_lin;
    real beta_step;
    real<lower=0, upper=1> l;
    
    vector[S] beta_lin_diff_norm;
    vector[S] beta_step_diff_norm;
    real<lower=0> std_beta_lin;
    real<lower=0> std_beta_step;
    
    vector[S] intercept;
    vector<lower=0>[S] stdy;
}

transformed parameters {
    vector[S] beta_lin_sub = beta_lin + beta_lin_diff_norm * std_beta_lin;
    vector[S] beta_step_sub = beta_step + beta_step_diff_norm * std_beta_step;
}

model {
    int n = 0;
    vector[Nall] mu_lin;
    vector[Nall] mu_step;
    vector[Nall] std;

    beta_lin ~ normal(0, 1);
    beta_step ~ normal(0, 1);
    l ~ beta(ab_beta, ab_beta);
    
    intercept ~ normal(0, 1);
    stdy ~ exponential(1);
    
    beta_lin_diff_norm ~ normal(0, 1);
    beta_step_diff_norm ~ normal(0, 1);
    std_beta_lin ~ normal(0, 1);
    std_beta_step ~ normal(0, 1);
    
    for (sub in 1:S) {
        mu_lin[n+1 : n+N[sub]] = beta_lin_sub[sub] * x[n+1 : n+N[sub]]
                                 + intercept[sub];
        mu_step[n+1 : n+N[sub]] = beta_step_sub[sub] * signx[n+1 : n+N[sub]]
                                 + intercept[sub];
        
        std[n+1 : n+N[sub]] = rep_vector(stdy[sub], N[sub]);
        
        n = n + N[sub];
    }
    
    target += log_mix(l, 
                      normal_lpdf(y | mu_lin, std),
                      normal_lpdf(y | mu_step, std));
}
"""

hierarchical_linear = """
data {
    int<lower=1> S;
    int<lower=0> N[S];
    int<lower=1> Nall;
    
    vector[Nall] x;
    vector[Nall] y;
}

transformed data {
    int<lower=0> Nsum[S+1];
    
    Nsum[1] = 0;
    for (sub in 1:S) {
        Nsum[sub+1] = Nsum[sub] + N[sub];
    }
}

parameters {
    real beta;
    
    vector[S] beta_diff_norm;
    real<lower=0> std_beta;
    
    vector[S] intercept;
    vector<lower=0>[S] stdy;
}

transformed parameters {
    vector[S] beta_sub = beta + beta_diff_norm * std_beta;
    vector[Nall] mu;
    vector<lower=0>[Nall] std;
    
    for (sub in 1:S) {
        mu[Nsum[sub]+1 : Nsum[sub+1]] = beta_sub[sub] 
                         * x[Nsum[sub]+1 : Nsum[sub+1]] 
                         + intercept[sub];
        
        std[Nsum[sub]+1 : Nsum[sub+1]] = rep_vector(stdy[sub], N[sub]);
    }
}

model {
    beta ~ normal(0, 1);
    
    beta_diff_norm ~ normal(0, 1);
    std_beta ~ exponential(1);
    
    intercept ~ normal(0, 1);
    stdy ~ exponential(1);
    
    y ~ normal(mu, std);
}
    
generated quantities {
    vector[Nall] log_lik;
    for (n in 1:Nall){
        log_lik[n] = normal_lpdf(y[n] | mu[n], std[n]);
    }
}
"""

model_code = hierarchical_linear

sm = pystan.StanModel(model_code=model_code)

with pd.HDFStore(fstore, mode='a', complevel=7, complib='blosc') as store:
    store['model'] = pd.Series(model_code)
    
    
#%% define function that returns data in the format that pystan wants
def get_stan_data(x, y, normalise, regressor, choices):
    if regressor == 'choice':
        choicesigns = np.sign(choices.loc[x.index])
        choicesigns.index = x.index
        x = x.abs() * choicesigns
        
    x_step = np.sign(x)
    
    if normalise == 'local':
        normfun = lambda s: (s - s.mean()) / s.std()
        
        x = x.groupby('subject').apply(normfun)
        x_step = x_step.groupby('subject').apply(normfun)
        y = y.groupby('subject').apply(normfun)
    
    return {'x_linear': x.values, 'x_step': x_step.values,
            'y': y.values,
            'S': y.index.get_level_values('subject').unique().size,
            'N': y.groupby('subject').count().values,
            'Nall': y.size}
    

#%% initialise output DataFrames
S = nsamples - nsamples // 2
samples = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, times, np.arange(S * nchains.sum(), dtype=int)], 
                names=['label', 'time', 'sample']), 
        columns=pd.MultiIndex.from_product(
                [['choice', 'dotx'], ['beta_lin', 'beta_step']], 
                names=['regressor', 'variable']), dtype=np.float64)
samples.sort_index(axis=0, inplace=True)
samples.sort_index(axis=1, inplace=True)

loodiffs = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, times], 
                names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [['choice', 'dotx', 'linear', 'step', 'ev_vs_resp'], 
                 ['diff', 'se']], 
                names=['model', 'measure']), dtype=np.float64)
loodiffs.sort_index(axis=0, inplace=True)
loodiffs.sort_index(axis=1, inplace=True)

sample_diagnostics = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, times], 
                names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [['choice', 'dotx'], ['beta_lin', 'beta_step'], ['neff', 'Rhat']], 
                names=['regressor', 'variable', 'measure']), 
        dtype=np.float64)
sample_diagnostics.sort_index(axis=0, inplace=True)
sample_diagnostics.sort_index(axis=1, inplace=True)


#%% run inference
def compute_loodiff(log_liks, use_mean=False):
    loos = []
    measure = []
    
    for model in log_liks:
        loosum_model, loos_model, ks = psis.psisloo(model)
        loos.append(loos_model)
        if use_mean:
            measure.append(loosum_model / loos_model.size)
        else:
            measure.append(loosum_model)
            
    if use_mean:
        return (measure[0] - measure[1], 
                (loos[0] - loos[1]).std() / np.sqrt(loos[0].size))
    else:
        return (measure[0] - measure[1], 
                np.sqrt( (loos[0] - loos[1]).var() * loos[0].size ))


for label in labels:
    trial_counts = []
    
    for time in times:
        print('\rlabel %s, time = % 5d' % (label, time), end='', flush=True)
        
        DMt = DM.xs(time, level='time')
        DMt = DMt.reorder_levels(['subject', 'trial', 'delay'])
        data = pd.DataFrame(
                np.c_[DMt.values, 
                      np.tile(alldata.xs(time, level='time')[label], len(delays))], 
                index=DMt.index, columns=[r_name, label])
        
        data.index = data.index.droplevel('delay')
        
        # remove trials which are > toolate to response
        if not resp_align:
            data = data.loc[rts[(time - rts) < toolate].index]
        
        data = data.dropna()
        
        # count remaining trials, ensure that subjects without any data have
        # 0 count
        trcnts = data[label].groupby('subject').count()
        ind = pd.Index(subjects).difference(trcnts.index)
        trcnts = pd.concat([trcnts, pd.Series(np.zeros(ind.size), index=ind)])
        trial_counts.append(trcnts.sort_index())
        
        # only use subjects with more than 30 trials
        data = data.loc[subjects[trial_counts[-1] > 30]]
        
        logliks = dict(choice_linear=None, choice_step=None,
                       dotx_linear=None, dotx_step=None)
        for regressor in ['dotx', 'choice']:
            standata = get_stan_data(data[r_name], data[label], normalise, 
                                     regressor, choices)
            
            # inference for models
            for modelname in ['linear', 'step']:
                llname = regressor + '_' + modelname
                if skip_mix and llname in ['dotx_step', 'choice_linear']:
                    continue
                
                varname = 'beta_lin' if modelname == 'linear' else 'beta_step'
                
                standata['x'] = standata['x_' + modelname]
                
                for c, nchain in enumerate(nchains):
                    fit = sm.sampling(standata, iter=nsamples, chains=nchain)
                
                    samples_tmp = fit.extract(['beta', 'log_lik'])
                    
                    samples.loc[
                            (label, time, slice(c * S * nchain, 
                                                (c + 1) * S * nchain - 1)), 
                            (regressor, varname)] = samples_tmp['beta']
                    
                    if c == 0:
                        logliks[llname] = samples_tmp['log_lik']
                    else:
                        logliks[llname] = np.r_[
                                logliks[llname], 
                                samples_tmp['log_lik']]
                
                if nchains.size == 1:
                    summ = pystan.misc._summary_sim(fit.sim, ['beta'], [0.5])
                    sample_diagnostics.loc[
                            (label, time), 
                            (regressor, varname, 'neff')] = summ['ess']
                    sample_diagnostics.loc[
                            (label, time), 
                            (regressor, varname, 'Rhat')] = summ['rhat']
        
        if not skip_mix:
            # compare linear vs. step models (negative supports step)
            for regressor in ['choice', 'dotx']:
                diff, se = compute_loodiff(
                        [logliks[regressor + '_linear'], 
                         logliks[regressor + '_step']], 
                        use_mean=use_loodiff_mean)
                
                loodiffs.loc[(label, time), (regressor, 'diff')] = diff
                loodiffs.loc[(label, time), (regressor, 'se')] = se
                
            # compare choice vs. dotx regressors (negative supports choice)
            for modelname in ['linear', 'step']:
                diff, se = compute_loodiff(
                        [logliks['dotx_' + modelname], 
                         logliks['choice_' + modelname]], 
                        use_mean=use_loodiff_mean)
                
                loodiffs.loc[(label, time), (modelname, 'diff')] = diff
                loodiffs.loc[(label, time), (modelname, 'se')] = se
            
        # compare linear dotx model (evidence) vs. step choice model (response)
        diff, se = compute_loodiff(
                [logliks['dotx_linear'], logliks['choice_step']], 
                use_mean=use_loodiff_mean)
        
        loodiffs.loc[(label, time), ('ev_vs_resp', 'diff')] = diff
        loodiffs.loc[(label, time), ('ev_vs_resp', 'se')] = se
                     
        try:
            # store in HDF-file, note that columns cannot be multi-index,
            # when you want to append to a table, hence use tmp-file
            with pd.HDFStore(fstore + '.tmp', mode='w', complib='blosc', 
                             complevel=7) as store:
                store['samples'] = samples
                store['loodiffs'] = loodiffs
                store['sample_diagnostics'] = sample_diagnostics
        except:
            warnings.warn("Couldn't store intermediate result, skipping.")

print('')

trial_counts = pd.concat(trial_counts, axis=1, keys=times).stack()

try:
    with pd.HDFStore(fstore, mode='r+', complevel=7, complib='blosc') as store:
        store['trial_counts'] = trial_counts
        store['samples'] = samples
        store['loodiffs'] = loodiffs
        store['sample_diagnostics'] = sample_diagnostics
except:
    warnings.warn('Results not saved yet!')
else:
    os.remove(fstore + '.tmp')

