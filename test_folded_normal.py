#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:04:23 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import pystan
import os
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

import source_statistics as ss
import helpers


#%% load source-level results
# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201707031206.h5'

r_name = 'dot_x'

file = os.path.join(ss.bem_dir, basefile[:-3] + '_slabs_%s.h5' % r_name)
second_level = pd.read_hdf(file, 'second_level_src')
options = pd.read_hdf(file, 'options')

testlabel = 'L_v23ab_ROI-lh'
testtimes = [120, 170, 320, 400]
testtheta = 0.5
N = 100

S = helpers.find_available_subjects(megdatadir=helpers.megdatadir).size


#%% create Stan model
folded_normal_stan = """
data {
 int<lower = 0> S;
 real y[S];
}

parameters {
  real<lower=0> mu;
  real<lower=0> sigma;
  real<lower=0, upper=1> theta;
}

model {
 sigma ~ normal(0, 5);
 mu ~ normal(0, 5);
 theta ~ beta(1, 1);
 
 for (s in 1:S)
     target += log_mix(theta,
                       normal_lpdf(y[s] |  mu, sigma),
                       normal_lpdf(y[s] | -mu, sigma));
}
"""

# pre-compile using random data
fit = pystan.stan(model_code=folded_normal_stan, 
                  data={'S': S, 'y': np.random.randn(S)},
                  iter=1000, chains=4)


#%% function that samples from folded normal mixture given parameters
def sample_fnm_data(mu, sigma, theta, size=10):
    if type(size) is int:
        size = (size,)
    assert type(size) is tuple
    
    data = np.random.randn(*size) * sigma
    signs = np.sign((np.random.rand(*size) < theta) - 0.5)
    
    return data + signs * mu


#%% run inference with fnm on sampled data
sl_cols = pd.Index(['mu_mean', 'mu_std', 'mu_t', 'mu_testval', 'mu_p_large', 
                    'sigma_mean', 'sigma_std', 'theta_mean', 'theta_std', 
                    'lp_mean', 'lp_std', 'overlap', 'consistency'], 
                   name='measure')
sl_ind = pd.MultiIndex.from_product([testtimes, np.arange(N)], 
                                     names=['testtime', 'rep'])
second_level_test = pd.DataFrame([], dtype=float, index=sl_ind, 
                                 columns=sl_cols)

for ttime in testtimes:
    print('testtime = %3d ...' % ttime)
    
    sl = second_level.loc[(testlabel, ttime)]
    for rep in range(N):
        data = sample_fnm_data(sl.mu_mean, sl.sigma_mean, testtheta, S)
        
        fit2 = pystan.stan(fit=fit, data={'S': S, 'y': data},
                           iter=1000, chains=4)

        samples = fit2.extract()

        ind = (ttime, rep)
        second_level_test.ix[ind, 'mu_mean'] = samples['mu'].mean()
        second_level_test.ix[ind, 'mu_std'] = samples['mu'].std()
        second_level_test.ix[ind, 'mu_t'] = (
                  second_level_test.ix[ind, 'mu_mean']
                / second_level_test.ix[ind, 'mu_std'])
        second_level_test.ix[ind, 'mu_testval'] = np.percentile(
                samples['mu'], options.alpha)
        second_level_test.ix[ind, 'mu_p_large'] = np.mean(
                samples['mu'] > options.p_thresh)
        second_level_test.ix[ind, 'sigma_mean'] = samples['sigma'].mean()
        second_level_test.ix[ind, 'sigma_std'] = samples['sigma'].std()
        second_level_test.ix[ind, 'theta_mean'] = samples['theta'].mean()
        second_level_test.ix[ind, 'theta_std'] = samples['theta'].std()
        second_level_test.ix[ind, 'lp_mean'] = samples['lp__'].mean()
        second_level_test.ix[ind, 'lp_std'] = samples['lp__'].std()
        second_level_test.ix[ind, 'overlap'] = (
                scipy.stats.norm.logcdf(
                        0, loc=second_level_test.ix[ind, 'mu_mean'], 
                        scale=second_level_test.ix[ind, 'sigma_mean']))
        second_level_test.ix[ind, 'consistency'] = (
                1 - np.mean(  (samples['theta'] > 0.25) 
                            & (samples['theta'] < 0.75)))
        
        
#%% compare results statistics to that of original finding
plot_measures = ['mu_mean', 'mu_t', 'mu_testval', 'mu_p_large', 'sigma_mean',
                 'theta_mean', 'consistency', 'overlap']
M = len(plot_measures)
C = int(np.ceil(M/2))

for ttime in testtimes:
    sl = second_level.loc[(testlabel, ttime)]
    
    fig, axes = plt.subplots(2, C)
    axes = axes.flatten()
    
    for ax, measure in zip(axes, plot_measures):
        sns.distplot(second_level_test.loc[(ttime, slice(None)), measure], 
                     kde=False, rug=True, ax=ax)
        ax.plot([sl[measure]]*2, [0, ax.get_ylim()[1]*0.1], 'r')
        
    axes[0].set_title('%s at %3d ms' % (testlabel, ttime))