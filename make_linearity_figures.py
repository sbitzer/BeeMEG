#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:45:31 2017

@author: bitzer
"""

import helpers
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


#%% load results (samples)
# resp-align, delay=400, times=[-700, 400], area 4
fname = 'source_linearity_201801041820.h5'

with pd.HDFStore(os.path.join(helpers.resultsdir, fname), 'r') as store:
    r_name = store['r_name'][0]
    options = store['scalar_params']
    samples = store['samples']
    loodiffs = store['loodiffs']
    sample_diagnostics = store['sample_diagnostics']

resp_align = options.resp_align

labels = samples.index.levels[0]
area = labels[0][2:-7]
times = samples.index.levels[1]


#%% plot some results
fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(7.5, 8))

fliplabels = {True: 'response-signed', False: 'x-coordinate'}
flipcols = {True: 'C0', False: 'C1'}

varlabels = {'beta': r'linear model population-level effect ($\beta$)', 
             'loo': 'evidence for linear'}

for var in varlabels.keys():
    if var == 'beta':
        row = axes[0]
    else:
        row = axes[1]
    
    for label, ax in zip(labels, row):
        if var == 'beta':
            subset = samples.loc[(label, slice(None)), (slice(None), 'beta_lin')]
            
            # get sample quantiles
            quantiles = subset.groupby('time').quantile([0.025, 0.5, 0.975])
            
            # plot 95% PPI bands
            for flip in [True, False]:
                ax.fill_between(
                        times, 
                        quantiles.loc[(slice(None), 0.025), flip].values.flatten(),
                        quantiles.loc[(slice(None), 0.975), flip].values.flatten(), 
                        facecolor=flipcols[flip], alpha=0.2)
                
            # plot medians
            for flip in [True, False]:
                ax.plot(times, quantiles.loc[(slice(None), 0.5), flip], lw=2,
                        color=flipcols[flip], label=fliplabels[flip])
        
        elif var == 'loo':
            # plot 2*SE bands
            for flip in [True, False]:
                subset = loodiffs.loc[(label, slice(None)), flip]
                ax.fill_between(
                        times, 
                        subset['diff'] - 2*subset['se'],
                        subset['diff'] + 2*subset['se'], 
                        facecolor=flipcols[flip], alpha=0.2)
                
            # plot loodiff
            for flip in [True, False]:
                subset = loodiffs.loc[(label, slice(None)), flip]
                ax.plot(times, subset['diff'], lw=2,
                        color=flipcols[flip], label=fliplabels[flip])
            
        # plot 0 as reference
        ax.plot(times[[0, -1]], [0, 0], '--k', lw=1, zorder=1)
            
    row[0].set_ylabel(varlabels[var])

for label, ax in zip(labels, axes[0]):
    ax.set_title(label[0] + '-' + area)
    
for label, ax in zip(labels, axes[1]):
    if resp_align:
        ax.set_xlabel('time from response (ms)')
    else:
        ax.set_xlabel('time from first dot onset (ms)')
        
# plot response indicator
if resp_align and times[-1] >= 0:
    for row in axes:
        yl = row[0].get_ylim()
        for ax in row:
            ax.plot([0, 0], yl, ':k', label='time of response', zorder=1)
        row[0].set_ylim(yl);

#axes[0, 0].set_ylim(-0.4, 0.4)

axes[0, 0].legend(loc='lower left')

fig.tight_layout()

figname = 'linearity_%s_%s_%s_delay%d.svg' % (
        r_name, 'response-aligned' if resp_align else 'firstdot-aligned',
        area, options.delay)
fig.savefig(os.path.join(helpers.figdir, figname), dpi=300)


#%% check whether beta_lin or beta_step are larger across several time points
tsplit = -170
tfirst = -2000
tlast = 110

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 4))

flipcols = {'original': 'C1', 'flipped': 'C0'}

ylabel = 'x-coordinate - response-signed (posterior mean beta)'

for label, ax in zip(labels, axes):
    subset = samples.loc[(label, slice(tfirst, tsplit), slice(None)),
                         (slice(None), 'beta_lin')]
    means = subset.groupby('time').mean()
    subset = samples.loc[(label, slice(tsplit+1, tlast), slice(None)),
                         (slice(None), 'beta_lin')]
    means = pd.concat([means, subset.groupby('time').mean()],
                      keys=['early', 'late'], names=['period', 'time'])
    means.columns = means.columns.levels[0].map(lambda x: fliplabels[x])
    
    diff = means[fliplabels[False]] - means[fliplabels[True]]
    diff.name = ylabel
    
    print(label)
    for period in ['early', 'late']:
        tval, pval = scipy.stats.ttest_1samp(diff[period], 0)
        print('%5s period: t = %+4.2f, p = %6.4f' % (period, tval, pval))
    
    diff = diff.reset_index(level='period')
    
    sns.boxplot(x='period', y=ylabel, data=diff, ax=ax, color='0.5', width=0.6,
                notch=True)
    
    ax.set_title(label[0] + '-' + area)
    
yl = axes[0].get_ylim()
axes[0].fill_between(ax.get_xlim(), [yl[1], yl[1]], 
                color=flipcols['original'], zorder=0, alpha=0.2)
axes[0].fill_between(ax.get_xlim(), [yl[0], yl[0]], 
                color=flipcols['flipped'], zorder=0, alpha=0.2)
axes[1].fill_between(ax.get_xlim(), [yl[1], yl[1]], 
                color=flipcols['flipped'], zorder=0, alpha=0.2)
axes[1].fill_between(ax.get_xlim(), [yl[0], yl[0]], 
                color=flipcols['original'], zorder=0, alpha=0.2)
axes[0].set_ylim(yl)

axes[1].set_ylabel('')
fig.tight_layout()

# svg didn't work with transparency in LibreOffice
figname = 'linearity_orig-flipped_%s_%s_%s_delay%d.png' % (
        r_name, 'response-aligned' if resp_align else 'firstdot-aligned',
        area, options.delay)
fig.savefig(os.path.join(helpers.figdir, figname), dpi=300)