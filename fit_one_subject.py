# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:54:58 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import random
import numpy as np
import pandas as pd
import os.path
from scipy.io import loadmat
import rtmodels
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob
import helpers

    
#%%--- most important user choices --------------------------------------------
datadir = 'data'

# subject index as coded in file names
sub = 15

# whether you want to use collapsing bounds
collapse = True

# which trials to use for inference and test?
# - there were 480 trials
# - in the order of the dotpos the first 240 trials differ from the second 240
#   trials only in that the sign of their x-positions is flipped
# - give here the indices of the trials in dotpos that you want to hold out
#   from inference to estimate a goodness of fit measure
# - give empty array to use all trials for inference
#testind = np.array([], dtype=np.int16)
testind = np.arange(400, 480, dtype=np.int16)

# number of samples to plot per distribution later, set to 0 for no plot
S = 0
# -----------------------------------------------------------------------------

#%% load data
mat = loadmat(os.path.join(datadir, 'batch_dots_pv2.mat'),
              variable_names=['xdots_n', 'ydots_n'])

# L - number of trials, S - maximum number of presented dots in one trial
L, D = mat['xdots_n'].shape
# last element is the target position for that trial, so ignore
D = D - 1
dotpos = np.full((D, 2, L), np.nan)
dotpos[:, 0, :] = mat['xdots_n'][:, :-1].T
dotpos[:, 1, :] = mat['ydots_n'][:, :-1].T

dotstd = 70.

respRT = helpers.load_subject_data(sub)

toresponse = helpers.defaultto
dotdt = 0.1

cond = helpers.defaultcond

feature_means = np.c_[[-cond, 0], [cond, 0]]

# sort trials such that fitind are first, because EP-ABC will only fit the 
# first couple of trials that are stored in the model
fitind = np.setdiff1d(np.arange(L), testind, assume_unique=True)
allind = np.r_[fitind, testind]
Trials = dotpos[:, :, allind]

# DDM-equiv: use this, if you don't want to use dot positions in the model
#Trials = respRT.loc[allind+1, 'stimulus']

#%% prepare model

# empty parameter container
pars = pyEPABC.parameters.parameter_container()

# common parameters
pars.add_param('ndtmean', -1, 1)
pars.add_param('ndtspread', -1.5, 1, exponential())
pars.add_param('noisestd', 4, 1.5, exponential())
pars.add_param('bound', 0, 1, gaussprob(0.5, 0.5))
pars.add_param('prior', 0, 1, gaussprob())
pars.add_param('lapseprob', -1, 1, gaussprob())
pars.add_param('lapsetoprob', 0, 1, gaussprob())

# collapsing bound parameters
if collapse:
    pars.add_param('bshape', np.log(1.4), 1.5, exponential())
    pars.add_param('bstretch', 0, 1, gaussprob())

# make model
model = rtmodels.discrete_static_gauss(maxrt=dotdt*D, choices=[-1, 1], 
    Trials=Trials, means=feature_means, intstd=dotstd)

model.dt = dotdt
model.toresponse = toresponse
# this is ignored when ndtmean is fitted, but makes a difference when 
# predicting from the model and not providing a value for ndtmean in the call
model.ndtmean = -10

# plot prior distribution of parameters
if S > 0:
    pars.plot_param_dist(S=S)

    
#%% setup and run EP-ABC
# wrapper for sampling from model and directly computing distances
simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
    data[0], data[1], dind, pars.transform(parsamples), pars.names)
distfun = None

# maximum distance allowed for accepting samples, note that for 
# response_dist this is in units of RT
epsilon = 0.05

# normalising constant of the uniform distribution defined by distfun and 
# epsilon; for response_dist this is: (2*epsilon for the Euclidean distance 
# between true and sampled RT and another factor of 2 for the two possible 
# responses - timed out responses don't occur here, because they are only a
# point mass, i.e., one particular value of response and RT)
veps = 2 * 2 * epsilon

data = respRT.loc[fitind+1, ['response', 'RT']].values

# run EPABC
ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
    data, simfun, distfun, pars.mu, pars.cov, epsilon=epsilon, minacc=2000, 
    samplestep=10000, samplemax=6000000, npass=2, alpha=0.5, veps=veps)
    
print('logml = %8.2f' % ep_logml)


#%% evaluate posterior
# sample from EPABC posterior (for all trials)
NP = 15000
choices, rts, samples_pos = model.gen_response_from_Gauss_posterior(
    np.arange(L), pars.names, ep_mean, ep_cov, NP, 
    pars.transform, return_samples=True)
samples_pos = pd.DataFrame(samples_pos, columns=pars.names)

# compute posterior predictive log-likelihoods
ppls = rtmodels.estimate_abc_loglik(respRT.loc[allind+1, 'response'], 
    respRT.loc[allind+1, 'RT'], choices, rts, epsilon)
print('sum of ppls = %8.2f' % ppls.sum())
print('sum of ppls for fitted trials = %8.2f' % ppls[:fitind.size].sum())
print('sum of ppls for test trials = %8.2f' % ppls[fitind.size:].sum())

# plot the posterior(s)
if S > 0:
    pars.plot_param_dist(ep_mean, ep_cov, S)
    
"""
Estimate the contribution of the nondecision time by computing the ratio of 
nondecision time to RT for each trial. This is only approximate, because I 
resample the nondecision times from the posterior samples.
"""
# for each sample from the posterior sample from the corresponding ndt distribution
ndt = np.zeros(NP)
for p in range(NP):
    ndt[p] = random.lognormvariate(samples_pos['ndtmean'][p], 
                                   samples_pos['ndtspread'][p])

print('median ratio of ndt to total RT: %4.2f' % np.median(ndt / rts))