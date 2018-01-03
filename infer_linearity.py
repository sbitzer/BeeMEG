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
import logging

# for switching off a pystan warning about sampling extraction
logger = logging.getLogger()
loglev = logger.getEffectiveLevel()


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

delay = 400

resp_align = True

# allows you to exclude trials based on response-aligned time in first dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = 5000

if resp_align:
    timeslice = [-700, delay]
    fbase = 'response-aligned'
else:
    timeslice = [delay, 1500]
    fbase = 'firstdot-aligned'

times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                 name='time')

fbase = 'linearity_%s_%s_norm%s_delay%3d_toolate%+05d' % (
            fbase, area, normalise, delay, toolate)
fstore = os.path.join(
        helpers.resultsdir, 
        pd.datetime.now().strftime('source_linearity'+'_%Y%m%d%H%M'+'.h5'))

r_colors = {'intercept': 'C0', 'dot_x_time': 'C1'}
r_labels = {'dot_x_time': 'x-coord',
            'intercept': 'intercept'}
r_names = list(r_labels.keys())
r_name = set(r_labels.keys()).difference({'intercept'}).pop()

ab_beta = 0.8

nsamples = 1000
nchains = 4

with pd.HDFStore(fstore, mode='w', complevel=7, complib='blosc') as store:
    store['scalar_params'] = pd.Series(
            [delay, resp_align, toolate, nsamples, nchains], 
            ['delay', 'resp_align', 'toolate', 'nsamples', 'nchains'])
    store['r_name'] = pd.Series(r_name)
    store['srcfile'] = pd.Series(srcfile)
    store['normalise'] = pd.Series(normalise)


#%% load data time courses
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

choices = regressors.subject_trial.response.loc[subjects]
rts = regressors.subject_trial.RT.loc[subjects] * 1000

alldata, epochtimes = helpers.load_area_tcs(
        srcfile, subjects, labels, resp_align, rts, normalise)


#%% load regressor time course
DM = subject_DM.get_trial_time_DM([r_name], epochtimes.values / 1000, 
                                  delay=delay / 1000, subjects=subjects)

DM.sort_index(axis=1, inplace=True)

if normalise == 'global':
    DM = subject_DM.normalise_DM(DM, True)

if resp_align:
    # exclude trials in which there was no response (timed out trials)
    def get_response_trials(sub):
        DMsub = DM.loc[sub]
        rtssub = rts.loc[sub]
        return DMsub.loc[(rtssub[rtssub != helpers.toresponse[1] * 1000].index, 
                          slice(None)), :]
    
    DM = pd.concat([get_response_trials(sub) for sub in subjects],
                    keys=subjects, names=['subject', 'trial', 'time'])
    
# now data and regressor are in the same order, so simply assign the 
# response-aligned index of alldata to the regressor values to get 
# response-aligned regressor values, if not response-aligned this transforms
# the time index from seconds to milliseconds
DM.index = alldata.index


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

sm = pystan.StanModel(model_code=hierarchical_linear)

with pd.HDFStore(fstore, mode='a', complevel=7, complib='blosc') as store:
    store['model'] = pd.Series(hierarchical_step_stan)
    
    
#%% define function that returns data in the format that pystan wants
def get_stan_data(x, y, normalise, choice_flip, choices):
    if choice_flip:
        x *= np.sign(x) * np.sign(choices.loc[x.index])
        
    if normalise == 'local':
        x = x.groupby('subject').apply(lambda s: (s - s.mean()) / s.std())
        y = y.groupby('subject').apply(lambda s: (s - s.mean()) / s.std())
    
    return {'x': x.values, 'signx': np.sign(x.values),
            'y': y.values,
            'S': y.index.get_level_values('subject').unique().size,
            'N': y.groupby('subject').count().values,
            'Nall': y.size,
            'ab_beta': ab_beta}
    

#%% initialise output DataFrames
samples = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, times, np.arange((nsamples - nsamples // 2) * nchains)], 
                names=['label', 'time', 'sample']), 
        columns=pd.MultiIndex.from_product(
                [[True, False], ['beta', 'l']], 
                names=['choice_flip', 'variable']), dtype=np.float64)
samples.sort_index(axis=1, inplace=True)


#%% run inference
for label in labels:
    trial_counts = []
    
    for time in times:
        print('\rlabel %s, time = % 5d' % (label, time), end='', flush=True)
        
        data = pd.concat([DM.xs(time, level='time'), 
                          alldata.xs(time, level='time')[label]], axis=1)
        
        # remove trials which are > toolate to response
        if not resp_align:
            data = data.loc[rts[(time - rts) < toolate].index]
        
        data = data.dropna()
        
        # count remaining trials
        trial_counts.append(data[label].groupby('subject').count())
        
        # only use subjects with more than 30 trials
        data = data.loc[subjects[trial_counts[-1] > 30]]
        
        for choice_flip in [False, True]:
            standata = get_stan_data(data[r_name], data[label], normalise, 
                                     choice_flip, choices)
            
            fit = sm.sampling(standata, iter=nsamples, chains=nchains)
            
            # wrapped into logging and warning calls to suppress a stupid 
            # pystan warning about ignoring dtypes when permuted=False and
            # a Python warning saying that they should use logger.warning 
            # instead of logger.warn
            logger.setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
            
                allsamples = fit.extract(permuted=False)
            
            logger.setLevel(loglev)
            
            for var in samples.columns.levels[1]:
                ind = fit.flatnames.index(var)
                
                samples.loc[(label, time, slice(None)), (choice_flip, var)] = (
                        allsamples[:, :, ind].flatten())
                
            try:
                # store in HDF-file, note that columns cannot be multi-index,
                # when you want to append to a table, hence use tmp-file
                with pd.HDFStore(fstore + '.tmp', mode='w', complib='blosc', 
                                 complevel=7) as store:
                    store['samples'] = samples
            except:
                warnings.warn("Couldn't store intermediate result, skipping.")

print('')

try:
    with pd.HDFStore(fstore, mode='r+', complevel=7, complib='blosc') as store:
        store['samples'] = samples
except:
    warnings.warn('Results not saved yet!')
else:
    os.remove(fstore + '.tmp')

trial_counts = pd.concat(trial_counts, axis=1, keys=times).stack()


#%% make some random test data
N = 2000
beta = 0.1
c = 0.06
sigma = 0.5

x = np.random.randn(N)

mu = beta * x
    
y = np.random.randn(N) * sigma + mu

data = {'N': N, 'x': x, 'signx': np.sign(x), 'y': y, 'ab_beta': 0.8}

#sns.regplot(x, y, x_estimator=np.mean, x_bins=8)
#
#
##%% run stan on test data
#fit = pystan.stan(model_code=step_stan, 
#                  data={'N': N, 'x': x, 'y': y},
#                  iter=1000, chains=4, verbose=True)
#
##fit.plot(['sigma', 'l', 'c', 'beta'])
#
##samples = fit.extract(['sigma', 'l', 'c', 'beta'])
#
#
##%% 
#mu = beta * np.sign(x)
#    
#y = np.random.randn(N) *sigma + mu
#
#fit2 = pystan.stan(fit=fit, data={'N': N, 'x': x, 'y': y},
#                   iter=1000, chains=4)
##fit2.plot(['sigma', 'l', 'c', 'beta'])
#
#
#%% make hierarchical random test data
S = 25
Nsub = np.array(np.random.randn(S) * 100, int)
Nsub *= -np.sign(Nsub)
Nsub += 300

# beta to 1 for strong effect, to 0.1 for weak effect, to 0 for no effect
beta = 0.1

beta_diff = np.random.randn(S) * 0.02
intercept = np.random.randn(S) * 0.1
sigmas = np.random.randn(S) * 0.1 + 1


#%%
def get_stan_data(beta, data=None):
    beta_sub = beta + beta_diff
    
    if data is None:
        data = {'S': S, 'N': Nsub, 'Nall': Nsub.sum()}
        
        x = []
        for sub in range(S):
            x.append(np.random.randn(Nsub[sub]))
            
        data['x_lin'] = np.concatenate(x)
        data['x_step'] = np.sign(data['x_lin'])
        data['x'] = data['x_lin']
            
    beta_all = []
    noise = []
    for sub in range(S):
        beta_all.append(np.full(Nsub[sub], beta_sub[sub]))
    
        noise.append(np.random.randn(Nsub[sub]) * sigmas[sub])
        
    beta_all = np.concatenate(beta_all)
    
    data['y_lin'] = np.concatenate(noise) + beta_all * data['x_lin']
    data['y_step'] = np.concatenate(noise) + beta_all * data['x_step']
    data['y'] = data['y_lin']
    
    return data


#%%
def compute_loodiff(log_liks):
    loos = []
    loosum = []
    for model in log_liks:
        loosum_model, loos_model, ks = psis.psisloo(model)
        loosum.append(loosum_model)
        loos.append(loos_model)
    
    return (loosum[0] - loosum[1], 
            np.sqrt( (loos[0] - loos[1]).var() * len(loos[0]) ))


def compare_lin_step(data, ytype='lin', iter=1000):
    data['y'] = data['y_' + ytype]
    
    data['x'] = data['x_lin']
    
    fit = sm.sampling(data, iter=iter)
    beta_lin_summary = fit.stansummary('beta')
    samples_lin = fit.extract(['beta', 'log_lik'])
    
    data['x'] = data['x_step']
    
    fit = sm.sampling(data, iter=iter)
    samples_step = fit.extract(['beta', 'log_lik'])
    
    diff, diffse = compute_loodiff([samples_lin['log_lik'], 
                                    samples_step['log_lik']])
        
    print('loo diff lin - step [-2*se, diff, +2*se]: [%5.2f, %5.2f, %5.2f]' % 
                     (diff - 2 * diffse, diff, diff + 2 * diffse))
    
    print('linear model:')
    print(beta_lin_summary)
    print('step model:')
    print(fit.stansummary('beta'))
#
#
#sns.regplot(data['x'], data['y'], x_estimator=np.mean, x_bins=8)
#
#def get_xy(xy, sub):
#    return data[xy][Nsub[:sub].sum():Nsub[:sub+1].sum()]
#
#
##%% reasonable initialisation to speed things up a little (speed-up effect may
## be little to non-existent)
#def get_hierarchical_init():
#    init = {}
#    
#    # intercept is within-subject mean
#    init['intercept'] = np.array([ysub.mean() for ysub in y])
#    
#    # stdy is within-subject std
#    init['stdy'] = np.array([ysub.std() for ysub in y])
#    
#    # randomise intercept using its standard error
#    init['intercept'] += np.random.randn(S) * init['stdy'] / np.sqrt(S-1)
#    
#    # randomise stdy using its across-subject std
#    init['stdy'] = np.abs(
#            init['stdy'] + np.random.randn(S) * init['stdy'].std())
#    
#    # l somewhere between 0 and 1
#    init['l'] = np.random.rand()
#    
#    # beta estimated from overall correlation (* 3, because the correlation is 
#    # the linear slope and the range of the normalised x should be roughly
#    # within [-3, 3])
#    est_beta = lambda x, y: np.abs(np.corrcoef(x, y)[0, 1]) * 3
#    init['beta'] = est_beta(data['y'], data['x'])
#    
#    # subject-specific beta-estimates
#    beta_sub = np.array([est_beta(xsub, ysub) for xsub, ysub in zip(x, y)])
#    
#    # std_beta as variability in betasub
#    init['std_beta'] = np.std(beta_sub - init['beta'])
#    
#    # beta_diff_norm estimated from difference between overall correlation and 
#    # subject-specific correlation
#    init['beta_diff_norm'] = (beta_sub - init['beta']) / init['std_beta']
#    
#    # randomise beta within across-subject variability of betas
#    init['beta'] = np.abs(init['beta'] + np.random.randn() * beta_sub.std())
#    
#    # randomise beta_diff_norm within its own variability
#    init['beta_diff_norm'] = np.abs(
#            init['beta_diff_norm'] + np.random.randn(S) * init['std_beta'])
#    
#    # randomise std_beta within its standard error as approximated from here:
#    # https://stats.stackexchange.com/questions/156518/what-is-the-standard-error-of-the-sample-standard-deviation
#    # assuming a normal distribution
#    init['std_beta'] = np.abs(
#            init['std_beta'] + np.random.randn() 
#            * np.sqrt(2 * init['std_beta'] ** 4 / (S-1)))
#    
#    return init
#
#
##%% 
#sm = pystan.StanModel(model_code=hierarchical_step_stan, verbose=True)
#
#fit = sm.sampling(data, iter=1000, chains=4, init=get_hierarchical_init)
#
#with open('fitres.txt', 'w') as file:
#    print(fit, file=file)