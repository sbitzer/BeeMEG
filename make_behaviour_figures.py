#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:21:58 2017

@author: bitzer
"""

import helpers
import regressors
import subject_DM
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% plot across-subject RT distribution
RTs = regressors.subject_trial.RT.copy()

# set timed out RTs to nan
RTs[RTs == 5.0] = np.nan
RTs.dropna(inplace=True)
RTs.sort_values(inplace=True)

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(7.5, 3))
sns.distplot(RTs, ax=axes[0])

axes[0].set_xlabel('response time (s)')
axes[0].set_ylabel('density')

axes[1].plot(RTs.values, (np.arange(RTs.size)+1) / RTs.size)
axes[1].set_xlabel('response time (s)')
axes[1].set_ylabel('cumulative density')

axes[1].set_xlim([RTs.min(), RTs.max()])

fig.subplots_adjust(left=0.1, bottom=0.17, right=0.97, top=0.91, wspace=0.3)

fig.savefig(os.path.join(figdir, 'pooledRTs.png'),
            dpi=300)


#%% plot evidence-choice correlations
dots = np.arange(1, 15)
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7.5, 3.5))

medianRT = regressors.subject_trial.loc[subjects, 'RT'].groupby(
        'subject').median().mean()

for coord, ax in zip(['x', 'y'], axes):
    r_names = ['dot_'+coord, 'sum_dot_'+coord, 'response']
    coord = coord + '-coordinate'
    DM = subject_DM.get_trial_DM(dots, subjects, r_names)
    
    def get_corrs(r_name):
        dmcols = [name for name in DM.columns if ((name == 'response') or 
                                                  (name.startswith(r_name)))]
        
        correlations = DM.groupby(level='subject').apply(
                lambda dm: dm[dmcols].corr().loc['response'])
        
        del correlations['response']
        correlations.columns = pd.Index(dots, name='dot')
        
        return correlations
    
    correlations = pd.concat([get_corrs(name) for name in r_names[:2]],
                             keys=['single dot', 'accumulated'], 
                             names=[coord, 'dot'], axis=1)
    correlations = correlations.stack().stack().reset_index()
    
    sns.stripplot('dot', 0, coord, data=correlations, jitter=True, 
                  dodge=True, ax=ax)
    
    ax.set_title(coord)

for ax in axes:
    # the first dot is shown at x-value 0 and the next dot at x=1, the x-axis
    # thus is time after first dot onset in 100 ms-units
    ax.set_autoscale_on(False)
    ax.plot(np.r_[1, 1] * medianRT * 10, ax.get_ylim(), ':k', label='median RT')

#ax.set_ylim([0, 1])
axes[0].set_ylabel('correlation with choice');
axes[0].legend().set_visible(False)
axes[1].set_ylabel('');
axes[1].legend()

fig.subplots_adjust(top=0.9, bottom=0.15, left=0.09, right=0.97, wspace=0.07)

fig.savefig(os.path.join(figdir, 'choice_correlations.png'),
            dpi=300)