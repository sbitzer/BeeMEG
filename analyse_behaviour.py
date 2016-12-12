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
    pars_basic = data['pars'][()]
    subjects = data['subjects']
    ep_means_basic = data['ep_means']
    ep_covs_basic = data['ep_covs']
    ppls_basic = data['ppls']
    ep_summary_basic = pd.DataFrame(data['ep_summary'], index=subjects, 
                                    columns=data['ep_summary_cols'])
    
with np.load(os.path.join(resultsdir, fname_collapse)) as data:
    # the "[()]" is necessary to get the model out of the numpy scalar array
    model = data['model'][()]
    testind = data['testind']
    pars_coll = data['pars'][()]
    ep_means_coll = data['ep_means']
    ep_covs_coll = data['ep_covs']
    ppls_coll = data['ppls']
    ep_summary_coll = pd.DataFrame(data['ep_summary'], index=subjects, 
                                   columns=data['ep_summary_cols'])

fitind = np.setdiff1d(np.arange(model.L), testind, assume_unique=True)
allind = np.r_[fitind, testind]


#%% load behavioural summary measures
allresp = helpers.load_all_responses()
resp_summary = pd.DataFrame({'accuracy': allresp['correct'].mean(level='subject'),
                             'medianRT': allresp['RT'].median(level='subject'),
                             'stdRT': allresp['RT'].std(level='subject')})

    
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
                                            'noisestd', 'medianRT', 'stdRT'])


#%% check response distribution of single subject
sub = 23
si = np.flatnonzero(ep_summary_coll.index == sub)[0]

# plot subject response distribution
chrt_sub = allresp.loc[sub, ['response', 'RT']]

plt.figure()
model.plot_response_distribution(chrt_sub.response, chrt_sub.RT)

# plot response distribution sampled from posterior
ch, rt = model.gen_response_from_Gauss_posterior(np.arange(model.L), 
    pars_coll.names, ep_means_coll[si, :], ep_covs_coll[si, :, :], 1, 
    transformfun=pars_coll.transform)

ax = model.plot_response_distribution(ch, rt, alpha=0.7)

chrt_m = pd.DataFrame(np.c_[ch, rt, ppls_coll[si, :]], index=allind+1, 
                      columns=['response_m', 'RT_m', 'ppls'])

# this will automatically align the responses according to trial
chrt = pd.concat([chrt_sub, chrt_m], axis=1)

# multiply RT by choice
chrt['resprt'] = chrt.response * chrt.RT
chrt['resprt_m'] = chrt.response_m * chrt.RT_m

# sort by response*RT
chrt.sort_values('resprt', inplace=True)

# plot
plt.figure()
plt.plot(chrt.loc[:, ['resprt', 'resprt_m']].values, '.')


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