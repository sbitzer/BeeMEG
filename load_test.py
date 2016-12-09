#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:24:28 2016

@author: bitzer
"""
import os
import numpy as np
import pandas as pd
import rtmodels
import helpers

megdatadir = os.path.expanduser('~/ZIH/gitrepos/BeeMEG/data/meg_behav')
resultsdir = os.path.expanduser('~/ZIH/gitrepos/BeeMEG/data/inf_results')
fname = 'infres_collapse_201612052119.npz'

with np.load(os.path.join(resultsdir, fname)) as data:
    # the "[()]" is necessary to get the model out of the numpy scalar array
    model = data['model'][()]
    pars = data['pars'][()]
    subjects = data['subjects']
    ep_means = data['ep_means']
    ep_covs = data['ep_covs']
    ep_summary = pd.DataFrame(data['ep_summary'], index=subjects, 
                              columns=data['ep_summary_cols'])
    NP = data['NP'][()]
    epsilon = data['epsilon']
    testind = data['testind']

toresponse = helpers.defaultto
    
fitind = np.setdiff1d(np.arange(model.L), testind, assume_unique=True)
allind = np.r_[fitind, testind]
    
si = 1

pg = pars.plot_param_dist(ep_means[si, :], ep_covs[si, :, :])

choices, rts = model.gen_response_from_Gauss_posterior(np.arange(model.L), 
    pars.names, ep_means[si, :], ep_covs[si, :, :], NP, pars.transform)

respRT = helpers.load_subject_data(subjects[si], behavdatadir=megdatadir)
ppls = rtmodels.estimate_abc_loglik(respRT.loc[allind+1, 'response'], 
    respRT.loc[allind+1, 'RT'], choices, rts, epsilon)

print('saved:    %-8.2f' % ep_summary.loc[subjects[si], 'pplsum'])
print('computed: %-8.2f' % ppls[:fitind.size].sum())

