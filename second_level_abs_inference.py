#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:54:14 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import pystan
import helpers


#%% load data
file = 'mne_subjects/fsaverage/bem/source_allsubs_201703301614.h5'
first_level_src = pd.read_hdf(file, 'first_level_src')

#N = first_level_src.shape[0]
#S = first_level_src.columns.levshape[0]

data = first_level_src.xs('dot_x', axis=1, level='regressor')
N, S = data.shape

#%%
N, S = data.shape
with pm.Model() as model:
    pi = pm.Beta('pi', alpha=1, beta=1, shape=N)
    mu = pm.HalfNormal('mu', sd=100, shape=N)
    sigma = pm.HalfCauchy('sigma', 5, shape=N)
    
    subjects = pm.NormalMixture('subjects', 
                                w=tt.tile(tt.stack([pi, 1-pi], axis=1), (S, 1)), 
                                mu=tt.tile(tt.stack([-mu, mu], axis=1), (S, 1)),
                                sd=tt.tile(tt.stack([sigma, sigma], axis=1), (S, 1)),
                                observed=data.values.flatten('F'))
    
    means, sds, elbos = pm.variational.advi(n=40000)
    
#%%
N, S = data.shape
with pm.Model() as model:
    pi = pm.Beta('pi', alpha=1, beta=1)
    mu = pm.HalfNormal('mu', sd=100)
    sigma = pm.HalfCauchy('sigma', 5)
    
    subjects = pm.NormalMixture('subjects', 
                                w=tt.stack([pi, 1-pi], axis=1), 
                                mu=tt.stack([-mu, mu], axis=1),
                                sd=tt.stack([sigma, sigma], axis=1),
                                observed=data.values[0, :])
    
    means, sds, elbos = pm.variational.advi(n=40000)
    
    
#%% random data in Stan
data = np.random.randn(1000, 34)
N, S = data.shape

folded_normal_stan = """
data {
 int<lower = 0> N;
 int<lower = 0> S;
 
 // choose matrix over array, because more efficient; matrices are
 // column-major so rows should be in the inner loop for maximum efficiency
 matrix[S, N] y;
}

parameters {
  real<lower=0> mu[N];
  real<lower=0> sigma[N];
  real<lower=0, upper=1> theta[N];
}

model {
 sigma ~ normal(0, 5);
 mu ~ normal(0, 5);
 theta ~ beta(5, 5);
 
 for (n in 1:N)
   for (s in 1:S)
     target += log_mix(theta[n],
                       normal_lpdf(y[s, n] |  mu[n], sigma[n]),
                       normal_lpdf(y[s, n] | -mu[n], sigma[n]));
}
"""

fit = pystan.stan(model_code=folded_normal_stan, 
                  data={'N': N, 'S': S, 'y': data.T},
                  iter=1000, chains=4)


#%% random data
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
                  data={'N': N, 'S': S, 'y': np.random.randn(S)},
                  iter=1000, chains=4)

r_name = 'dot_x'
fits = []
start = pd.datetime.now()
for ind in range(10):
    data = first_level_src.iloc[ind].xs(r_name, level='regressor').dropna()
    S = data.size
    
    fits.append(pystan.stan(fit=fit, data={'N': N, 'S': S, 'y': data},
                            iter=1000, chains=4))
end = pd.datetime.now()
print('elpased time for %d passes: %.2f s' % (ind+1, (end-start).total_seconds()))