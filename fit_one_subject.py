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


def load_subject_data(subject_index, datadir='data'):
    mat = loadmat(os.path.join(datadir, '%02d-meg.mat' % subject_index), 
                  variable_names=['m_dat'])
    
    extract_data = lambda mat: np.c_[np.abs(mat[:, 0]), np.sign(mat[:, 0]),
        mat[:, 5], mat[:, 3] / 10000]
    
    colnames = ['condition', 'stimulus', 'response', 'RT']
    
    # make a data frame with the relevant data
    respRT = pd.DataFrame(extract_data(mat['m_dat']), 
                          index=mat['m_dat'][:, 1], 
                          columns=colnames)
    
    toresponse = np.array([0, 5.0])

    # set RT of timed out responses to 5s
    respRT['RT'][respRT['response'] == toresponse[0]] = toresponse[1]
    
    # recode timed out responses (=0) as 1.5: this makes them 0 after the next
    # transformation to -1 and 1
    respRT.replace({'response': {toresponse[0]: 1.5}}, inplace=True)
    
    # unify coding of left stimuli (1 = -1) and right stimuli (2 = 1)
    respRT['response'] = (respRT['response'] - 1.5)  * 2    
    
    return respRT, toresponse
    
#%%--- most important user choices --------------------------------------------
datadir = 'data'

# subject index as coded in file names
sub = 3

# whether you want to use collapsing bounds
collapse = False

# indices of trials on which to fit; must be in [1, 480]; 
# 1-240 is data1, 241-480 is data2
fitind = np.arange(1, 241, dtype=np.int)
fitind = np.arange(1, 401, dtype=np.int)

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

respRT, toresponse = load_subject_data(sub, os.path.join(datadir, 'meg_behav'))
dotdt = 0.1

cond = 25

# check that all trials belong to that condition
assert(np.all(respRT['condition'] == cond))

feature_means = np.c_[[-cond, 0], [cond, 0]]

# sort respRT such that fitind are first, because EP-ABC will only fit the 
# first couple of trials that are stored in the model
testind = np.setdiff1d(np.arange(1, L+1), fitind, assume_unique=True)
respRT = pd.concat([respRT.loc[fitind], respRT.loc[testind]])

# subtract 1 because indeces in respRT start at 1, not at 0
Trials = dotpos[:, :, respRT.index - 1]
# DDM-equiv: use this, if you don't want to use dot positions in the model
#Trials = respRT['stimulus']

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

# run EPABC
ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
    np.c_[respRT.loc[fitind]['response'], respRT.loc[fitind]['RT']], simfun, distfun, 
    pars.mu, pars.cov, epsilon=epsilon, minacc=1000, samplestep=10000, 
    samplemax=4000000, npass=2, alpha=0.5, veps=veps)
    
print('logml = %8.2f' % ep_logml)


#%% evaluate posterior
# sample from EPABC posterior (for all trials)
NP = 15000
choices, rts, samples_pos = model.gen_response_from_Gauss_posterior(
    np.arange(L), pars.names, ep_mean, ep_cov, NP, 
    pars.transform, return_samples=True)
samples_pos = pd.DataFrame(samples_pos, columns=pars.names)

# compute posterior predictive log-likelihoods
ppls = rtmodels.estimate_abc_loglik(respRT['response'], 
    respRT['RT'], choices, rts, epsilon)
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