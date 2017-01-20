#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script for inferring rtmodels from the data of the BeeMEG experiment

Created on Mon Dec  5 10:54:42 2016

@author: bitzer
"""
import os
import numpy as np
import pandas as pd
import rtmodels
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob
import datetime
import traceback
import helpers


#%% basic choices and setup

# which models to fit?
models = ('basic', 'collapse')

# where's the data?
behavdatadir = helpers.behavdatadir

# where should results be?
resultsdir = helpers.resultsdir

# which subjects to fit this time (either None for all or a list of numbers)
subjects = None

# which trials to use for inference and test?
# - there were 480 trials
# - in the order of the dotpos the first 240 trials differ from the second 240
#   trials only in that the sign of their x-positions is flipped
# - give here the indices of the trials in dotpos that you want to hold out
#   from inference to estimate a goodness of fit measure
# - give empty array to use all trials for inference
#testind = np.array([], dtype=np.int16)
testind = np.arange(400, 480, dtype=np.int16)


#%% EP-ABC choices

# maximum distance allowed for accepting samples, 
# this is in units of RT (seconds)
epsilon = 0.05

# number of samples that need to be accepted before re-estimating posterior
minacc = 2000

# maximum number of samples before re-estimating posterior even when minacc is
# not reached yet
samplemax = 6000000

# number of passes through data
npass = 2

# how smooth should updates be?
# alpha=1 for full updates, 0<alpha<1 for partial/smooth updates 
alpha = 0.5

# number of samples to estimate posterior predictive likelihoods
NP = 15000


#%% pre-process and set experiment constants

# get subject indeces
if subjects is None:
    subjects = helpers.find_available_subjects(behavdatadir)
            
subjects = np.sort(np.array(subjects))
S = len(subjects)

dotpos = helpers.load_dots()
D, _, L = dotpos.shape

dotstd = helpers.dotstd
dotdt = helpers.dotdt

cond = helpers.cond

feature_means = np.c_[[-cond, 0], [cond, 0]]

toresponse = helpers.toresponse

# sort trials such that fitind are first, because EP-ABC will only fit the 
# first couple of trials that are stored in the model
fitind = np.setdiff1d(np.arange(L), testind, assume_unique=True)
allind = np.r_[fitind, testind]
Trials = dotpos[:, :, allind]


#%% make basic model and parameters
model = rtmodels.discrete_static_gauss(dt=dotdt, maxrt=dotdt*D, 
                                       toresponse=toresponse, choices=[-1, 1], 
                                       Trials=Trials, means=feature_means, 
                                       intstd=dotstd)


#%% setup EP-ABC

# normalising constant of the uniform distribution defined by distfun and 
# epsilon; for response_dist this is: (2*epsilon for the Euclidean distance 
# between true and sampled RT and another factor of 2 for the two possible 
# responses - timed out responses don't occur here, because they are only a
# point mass, i.e., one particular value of response and RT)
veps = 2 * 2 * epsilon


#%% inference
for mname in models:
    # generate time-stamped file name
    fname = datetime.datetime.now().strftime('infres_'+mname+'_%Y%m%d%H%M')
    
    # prepare common model parameters
    pars = pyEPABC.parameters.parameter_container()
    pars.add_param('ndtmean', -1, 1)
    pars.add_param('ndtspread', -1.5, 1, exponential())
    pars.add_param('noisestd', 4, 1.5, exponential())
    pars.add_param('bound', 0, 1, gaussprob(0.5, 0.5))
    pars.add_param('prior', 0, 0.5, gaussprob())
    pars.add_param('lapseprob', -1, 1, gaussprob())
    pars.add_param('lapsetoprob', 0, 1, gaussprob())
    
    # model-specific preparations
    if mname == 'basic':
        pass
    elif mname == 'collapse':
        pars.add_param('bshape', np.log(1.4), 1.5, exponential())
        pars.add_param('bstretch', 0, 1, gaussprob())
    else:
        raise ValueError('unknown model requested')
    
    # define function which simulates from model and evaluates result
    simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
        data[0], data[1], dind, pars.transform(parsamples), pars.names)
    
    # different results containers
    errors = {}
    ep_means = np.zeros((S, pars.P))
    ep_covs = np.zeros((S, pars.P, pars.P))
    ppls = np.zeros((S, L))
    ppls_N = np.zeros((S, L))
    ep_summary = pd.DataFrame([], index=subjects, columns=(['logml', 'pplsum',
                              'pplsum_test', 'nsamples', 'runtime', 'ndtmode',
                              'dtmode', 'ndt_vs_RT'] + list(pars.names)))
    for si, sub in enumerate(subjects):
        # load data
        respRT = helpers.load_subject_data(sub, behavdatadir)
        
        # data trials need to be in the order corresponding to the one in the 
        # model, remember that indeces in respRT start at 1, not 0
        data = respRT.loc[fitind+1, ['response', 'RT']].values

        #%% infer with EP-ABC and collect results
        print('subject: %02d, model: %s' % (sub, mname))
        
        # the for-loop is a fail-safe procedure: try to run EP-ABC 3 times when
        # it unexpectedly breaks, if it didn't run through the 3rd time, just 
        # continue with the next subject
        for run in range(3):
            try:
                ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
                    data, simfun, None, pars.mu, pars.cov, epsilon=epsilon, 
                    minacc=minacc, samplestep=10000, samplemax=samplemax, npass=npass, 
                    alpha=alpha, veps=veps)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                errors[sub] = 'run %d:\n' % run + traceback.format_exc()
            else:
                ep_means[si, :] = ep_mean
                ep_covs[si, :, :] = ep_cov
                ep_summary.loc[sub, 'logml'] = ep_logml
                ep_summary.loc[sub, 'nsamples'] = ntotal.sum()
                ep_summary.loc[sub, 'runtime'] = runtime.total_seconds()
                ep_summary.loc[sub, pars.names] = pars.get_transformed_mode(
                    ep_mean, ep_cov)
                
                #%% preliminary extra analysis
                
                ndtmode = np.exp(ep_summary.loc[sub, 'ndtmean'] - 
                                 ep_summary.loc[sub, 'ndtspread']**2)
                ep_summary.loc[sub, 'ndtmode'] = ndtmode
                # estimate decision time mode in relation to median RT
                ep_summary.loc[sub, 'dtmode'] = respRT['RT'].median() - ndtmode
                # compare non-decision time mode to median RT
                ep_summary.loc[sub, 'ndt_vs_RT'] = ndtmode / respRT['RT'].median()
                
                # compute posterior predictive log-likelihoods
                ppls[si, :], ppls_N[si, :] = pyEPABC.estimate_predlik(
                    respRT.loc[allind+1, ['response', 'RT']].values, simfun, 
                    ep_mean, ep_cov, epsilon, samplestep=NP)
                
                ep_summary.loc[sub, 'pplsum'] = ppls[si, :fitind.size].sum()
                ep_summary.loc[sub, 'pplsum_test'] = ppls[si, fitind.size:].sum()
                
                # breaks out of the for-loop when everything worked out
                break
        
        #%% save results
        np.savez_compressed(os.path.join(resultsdir, fname), subjects=subjects,
                            testind=testind, epsilon=epsilon, minacc=minacc,
                            samplemax=samplemax, npass=npass, alpha=alpha,
                            NP=NP, model=model, pars=pars, ep_means=ep_means,
                            ep_covs=ep_covs, ppls=ppls, ppls_N=ppls_N, 
                            ep_summary=ep_summary, 
                            ep_summary_cols=ep_summary.columns, errors=errors)