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
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os

figdir = helpers.figdir

subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)


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


#%% scatter plot of accuracy and RT across participants
axgrid = sns.jointplot('accuracy', 'medianRT', 
                       data=regressors.subject.loc[subjects])
axgrid.ax_joint.set_ylabel('median RT (s)')

axgrid.fig.savefig(os.path.join(figdir, 'subject_behaviour.png'), dpi=300)
axgrid.fig.savefig(os.path.join(figdir, 'subject_behaviour.eps'))


#%% plot evidence-choice correlations
dots = np.arange(1, 15)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7.5, 3.5))

medianRT = regressors.subject_trial.loc[subjects, 'RT'].groupby(
        'subject').median().mean()

for title, coord, ax in zip(['evidence', 'y-coordinate'], ['x', 'y'], axes):
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
                             keys=['momentary', 'accumulated'], 
                             names=[coord, 'dot'], axis=1)
    correlations = correlations.stack().stack().reset_index()
    
    sns.stripplot('dot', 0, coord, data=correlations, jitter=True, 
                  dodge=True, ax=ax)
    
    ax.set_title(title)

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


#%% plot correlation between dot_x and sum_dot_x
dots = np.arange(1, 15)
D = dots.size

correct = regressors.trial.correct
choice = regressors.subject_trial.loc[subjects, 'response']
dot_x = regressors.trial_dot.loc[(slice(None), dots), 'dot_x']
sum_x = regressors.trial_dot.loc[(slice(None), dots), 'sum_dot_x']

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# correlation with dot_x
ax = axes[0]
dotcorrs = pd.DataFrame(
        np.corrcoef(dot_x.unstack('trial'), sum_x.unstack('trial'))[:D, D:],
        index=pd.Index(dots, name='xcoord'), 
        columns=pd.Index(dots, name='sumx'))

corrim = ax.imshow(dotcorrs, 'PiYG', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(np.arange(D)); 
ax.set_xticklabels(dots)
ax.set_xlabel('accumulated evidence up to dot ...')
ax.set_yticks(np.arange(D)); 
ax.set_yticklabels(dots)
ax.set_ylabel('momentary evidence for dot ...')
cbar = fig.colorbar(corrim, ax=ax)
cbar.set_label('correlation')

# correlation with last added dot_x, correct response, choice
ax = axes[1]

choicecorrs = pd.DataFrame(
        [np.corrcoef(sum_x.unstack('trial'), choice.loc[sub])[-1, :-1]
         for sub in subjects], index=subjects, 
        columns=pd.Index(dots, name='dot'))
sns.boxplot(data=choicecorrs, ax=ax, color='C0')
lbox = mlines.Line2D([], [], color='C0', lw=3, label='choice')

l1, = ax.plot(np.diag(dotcorrs), color='C2', label='mom. ev.', lw=3, zorder=9)

corrcorrs = pd.Series(
        np.corrcoef(sum_x.unstack('trial'), correct)[-1, :-1],
        index=pd.Index(dots, name='correct'))
l2, = ax.plot(corrcorrs.values, color='C1', label='correct', lw=3, zorder=10)

ax.legend(handles=[lbox, l1, l2])
ax.set_xlabel('accumulated evidence up to dot ...')

fig.tight_layout()

fig.text(0.015, 0.92, 'A', weight='bold', fontsize=20)
fig.text(0.5, 0.92, 'B', weight='bold', fontsize=20)

fig.savefig(os.path.join(figdir, 'x-sumx-correlation.png'), dpi=300)