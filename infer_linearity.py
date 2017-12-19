#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:50:59 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import pystan
import matplotlib.pyplot as plt
#import seaborn as sns

#%% 
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


#%% define basic model
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

#%% make some random test data
N = 2000
beta = 0.1
c = 0.06
sigma = 0.5

x = np.random.randn(N)

mu = beta / 3 * x
    
y = np.random.randn(N) *sigma + mu

#sns.regplot(x, y, x_estimator=np.mean, x_bins=8)


#%% run stan on test data
fit = pystan.stan(model_code=step_stan, 
                  data={'N': N, 'x': x, 'y': y},
                  iter=1000, chains=4, verbose=True)

#fit.plot(['sigma', 'l', 'c', 'beta'])

#samples = fit.extract(['sigma', 'l', 'c', 'beta'])


#%% 
mu = beta * np.sign(x)
    
y = np.random.randn(N) *sigma + mu

fit2 = pystan.stan(fit=fit, data={'N': N, 'x': x, 'y': y},
                   iter=1000, chains=4)
#fit2.plot(['sigma', 'l', 'c', 'beta'])


#%% make hierarchical random test data
S = 34
Nsub = np.array(np.random.randn(S) * 100, int)
Nsub *= -np.sign(Nsub)
Nsub += 480

beta = 0.1
c = 0.06

beta_sub = beta + np.random.randn(S) * 0.03
intercept = c + np.random.randn(S) * 0.1
sigmas = np.random.randn(S) * 0.1 + 1

x = []
y = []
for sub in range(S):
    x.append(np.random.randn(Nsub[sub]))
    
    mu = beta_sub[sub] * np.sign(x[-1])
    
    y.append(np.random.randn(Nsub[sub]) * sigmas[sub] + mu)

ylin = []
for sub in range(S):
    mu = beta_sub[sub] / 3 * x[sub]
    
    ylin.append(np.random.randn(Nsub[sub]) * sigmas[sub] + mu)

data = {'S': S, 'N': Nsub, 'Nall': Nsub.sum(),
        'x': np.concatenate(x), 'y': np.concatenate(y)}


sns.regplot(data['x'], data['y'], x_estimator=np.mean, x_bins=8)

def get_xy(xy, sub):
    return data[xy][Nsub[:sub].sum():Nsub[:sub+1].sum()]


#%% reasonable initialisation to speed things up a little (speed-up effect may
# be little to non-existent)
def get_hierarchical_init():
    init = {}
    
    # intercept is within-subject mean
    init['intercept'] = np.array([ysub.mean() for ysub in y])
    
    # stdy is within-subject std
    init['stdy'] = np.array([ysub.std() for ysub in y])
    
    # randomise intercept using its standard error
    init['intercept'] += np.random.randn(S) * init['stdy'] / np.sqrt(S-1)
    
    # randomise stdy using its across-subject std
    init['stdy'] = np.abs(
            init['stdy'] + np.random.randn(S) * init['stdy'].std())
    
    # l somewhere between 0 and 1
    init['l'] = np.random.rand()
    
    # beta estimated from overall correlation (* 3, because the correlation is 
    # the linear slope and the range of the normalised x should be roughly
    # within [-3, 3])
    est_beta = lambda x, y: np.abs(np.corrcoef(x, y)[0, 1]) * 3
    init['beta'] = est_beta(data['y'], data['x'])
    
    # subject-specific beta-estimates
    beta_sub = np.array([est_beta(xsub, ysub) for xsub, ysub in zip(x, y)])
    
    # std_beta as variability in betasub
    init['std_beta'] = np.std(beta_sub - init['beta'])
    
    # beta_diff_norm estimated from difference between overall correlation and 
    # subject-specific correlation
    init['beta_diff_norm'] = (beta_sub - init['beta']) / init['std_beta']
    
    # randomise beta within across-subject variability of betas
    init['beta'] = np.abs(init['beta'] + np.random.randn() * beta_sub.std())
    
    # randomise beta_diff_norm within its own variability
    init['beta_diff_norm'] = np.abs(
            init['beta_diff_norm'] + np.random.randn(S) * init['std_beta'])
    
    # randomise std_beta within its standard error as approximated from here:
    # https://stats.stackexchange.com/questions/156518/what-is-the-standard-error-of-the-sample-standard-deviation
    # assuming a normal distribution
    init['std_beta'] = np.abs(
            init['std_beta'] + np.random.randn() 
            * np.sqrt(2 * init['std_beta'] ** 4 / (S-1)))
    
    return init


#%% 
sm = pystan.StanModel(model_code=hierarchical_step_stan, verbose=True)

fit = sm.sampling(data, iter=1000, chains=4, init=get_hierarchical_init)

with open('fitres.txt', 'w') as file:
    print(fit, file=file)