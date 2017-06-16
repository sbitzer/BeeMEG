#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:54:14 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import pystan
import os
import scipy.stats


#%% options
# store of first_level results
#basefile = 'source_allsubs_201703301614.h5'

# baseline [-0.3, 0], trialregs_dot=5
#basefile = 'source_HCPMMP1_allsubs_201703301614.h5'
# baseline None, trialregs_dot=5
#basefile = 'source_HCPMMP1_allsubs_201706091054.h5'
# baseline (-0.3, 0), trialregs_dot=5, GLM in source space
basefile = 'source_sequential_201706141650.h5'

directory = 'mne_subjects/fsaverage/bem/'

# regressors for which to infer across-subject strength
r_names = ['dot_x', 'accev', 'dot_y', 'abs_dot_x', 'abs_dot_y', 'response', 'entropy',
           'trial_time', 'intercept', 'accsur_pca', 'dot_x_cflip', 'accev_cflip']

# threshold for "posterior probability of the existence of a medium sized 
# effect", i.e., the probability that a sample from the posterior is > p_thresh
p_thresh = 0.5

# chunksize: store results every CS iterations
CS = 5000


#%% load data
#first_level_src = pd.read_hdf(os.path.join(directory, basefile), 
#                              'first_level_src')
first_level_src = pd.read_hdf('/media/bitzer/Data/source_sequential_201706141650.h5.tmp',
                              'first_level')
first_level_src = first_level_src.xs(0, level='permnr').xs('beta', level='measure', axis=1)

N = first_level_src.shape[0]
S = first_level_src.columns.levshape[0]

lenlabels = 0
if type(first_level_src.index.levels[0][0]) == str:
    lenlabels = max([len(l) for l in first_level_src.index.levels[0]])

chunks = [slice(c*CS, min((c+1)*CS, N)) for c in range(int(np.ceil(N/CS)))]


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


#%% run inference across all data
sl_cols = pd.Index(['mu_mean', 'mu_std', 'mu_t', 'mu_p_large', 
                    'sigma_mean', 'sigma_std', 'theta_mean', 'theta_std', 
                    'lp_mean', 'lp_std', 'overlap', 'consistency'], 
                   name='measure')

for r_name in r_names:
    print('%s\n%s' % (r_name, '-'*len(r_name)))
    file = os.path.join(directory, '{}_slabs_{}.h5'.format(basefile[:-3], r_name))
    
    r_data = first_level_src.xs(r_name, axis=1, level='regressor')
    
    with pd.HDFStore(file, mode='w', complib='blosc', complevel=7) as store:
        store['options'] = pd.Series({'p_thresh': p_thresh})
        
        for chunk in chunks:
            print('processing index %d to %d ...' % 
                  (chunk.start, chunk.stop-1), flush=True)
            
            start = pd.datetime.now()
            
            c_data = r_data.iloc[chunk]
            
            # create store data frame
            second_level_src = pd.DataFrame([], dtype=float, index=c_data.index, 
                                            columns=sl_cols)
            
            for ind in range(c_data.shape[0]):
                data = c_data.iloc[ind].dropna().values
                
                # some vertices do not exist for some subjects, so number of subjects varies
                Sd = data.size
                
                fit2 = pystan.stan(fit=fit, data={'S': Sd, 'y': data},
                                   iter=1000, chains=4)
                samples = fit2.extract()
                
                second_level_src.ix[ind, 'mu_mean'] = samples['mu'].mean()
                second_level_src.ix[ind, 'mu_std'] = samples['mu'].std()
                second_level_src.ix[ind, 'mu_t'] = (
                          second_level_src.ix[ind, 'mu_mean']
                        / second_level_src.ix[ind, 'mu_std'])
                second_level_src.ix[ind, 'mu_p_large'] = np.mean(
                        samples['mu'] > p_thresh)
                second_level_src.ix[ind, 'sigma_mean'] = samples['sigma'].mean()
                second_level_src.ix[ind, 'sigma_std'] = samples['sigma'].std()
                second_level_src.ix[ind, 'theta_mean'] = samples['theta'].mean()
                second_level_src.ix[ind, 'theta_std'] = samples['theta'].std()
                second_level_src.ix[ind, 'lp_mean'] = samples['lp__'].mean()
                second_level_src.ix[ind, 'lp_std'] = samples['lp__'].std()
                second_level_src.ix[ind, 'overlap'] = (
                        scipy.stats.norm.logcdf(
                                0, loc=second_level_src.ix[ind, 'mu_mean'], 
                                scale=second_level_src.ix[ind, 'mu_std']))
                second_level_src.ix[ind, 'consistency'] = (
                        1 - np.mean(  (samples['theta'] > 0.25) 
                                    & (samples['theta'] < 0.75)))
                
            end = pd.datetime.now()
            print('elpased time for this chunk: %.2f min' %
                  ((end-start).total_seconds() / 60, ))
            
            if lenlabels:
                store.append('second_level_src', second_level_src, 
                             min_itemsize={'label': lenlabels})
            else:
                store.append('second_level_src', second_level_src)
            
        