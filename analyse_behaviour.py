#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:20:28 2016

@author: bitzer
"""

import os
import numpy as np
import pandas as pd
import helpers
import seaborn as sns
import matplotlib.pyplot as plt


#%% load inference results
resultsdir = os.path.expanduser('~/ZIH/gitrepos/BeeMEG/data/inf_results')
#fname_collapse = 'infres_collapse_201612052119.npz'
#fname_basic = 'infres_basic_201612051900.npz'
fname_collapse = 'infres_collapse_201612061534.npz'
fname_basic = 'infres_basic_201612061157.npz'

with np.load(os.path.join(resultsdir, fname_basic)) as data:
    # the "[()]" is necessary to get the model out of the numpy scalar array
    model = data['model'][()]
    pars_basic = data['pars'][()]
    subjects = data['subjects']
    ep_means_basic = data['ep_means']
    ep_covs_basic = data['ep_covs']
    ep_summary_basic = pd.DataFrame(data['ep_summary'], index=subjects, 
                                    columns=data['ep_summary_cols'])
    
with np.load(os.path.join(resultsdir, fname_collapse)) as data:
    # the "[()]" is necessary to get the model out of the numpy scalar array
    model = data['model'][()]
    pars_coll = data['pars'][()]
    ep_means_coll = data['ep_means']
    ep_covs_coll = data['ep_covs']
    ep_summary_coll = pd.DataFrame(data['ep_summary'], index=subjects, 
                                   columns=data['ep_summary_cols'])
    

#%% load behavioural summary measures
allresp = helpers.load_all_responses()
resp_summary = pd.DataFrame({'accuracy': allresp['correct'].mean(level='subject'),
                             'medianRT': allresp['RT'].median(level='subject')})

    
#%% basic or collapsing bound?
measure = 'logml'

datax = ep_summary_basic[measure]
datax.name = 'basic ('+measure+')'
datay = ep_summary_coll[measure]
datay.name = 'collapse ('+measure+')'

jg = sns.jointplot(datax, datay)
jg.ax_joint.plot(jg.ax_joint.get_xlim(), jg.ax_joint.get_xlim())


#%% check correlations
# put everything together in one dataframe for easier handling
allinfo = pd.concat([ep_summary_coll, resp_summary], axis=1)
allinfo['hue'] = allinfo['ndt_vs_RT'] < 0.66

pg = sns.pairplot(allinfo, hue='hue', vars=['accuracy', 'ndt_vs_RT', 'logml',
                                            'pplsum_test', 'noisestd'])


#%% sequential effects?
# these RTs will be sorted by their trial index so temporal structure within
# the experiment will be lost, but the subjects are aligned according to the
# trials with the same dot positions
RTs = allresp['RT'].reset_index(level='subject').pivot(columns='subject')

RTs = pd.DataFrame([])
for sub in subjects:
    resp = allresp.loc[sub]
    RTs[sub] = resp['RT'].values
    
plt.figure()
plt.plot((RTs - RTs.median()).mean())