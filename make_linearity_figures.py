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
from matplotlib.gridspec import GridSpec
import seaborn as sns
import scipy.stats


#%% load results (samples)
# resp-align, delay=400, times=[-700, 400], area 4, only linear vs step
#fname = 'source_linearity_201801041820.h5'

# resp-align, delay=400, times=[-500, 400], area 4
fname = 'source_linearity_201804111716.h5'

with pd.HDFStore(os.path.join(helpers.resultsdir, fname), 'r') as store:
    r_name = store['r_name'][0]
    options = store['scalar_params']
    samples = store['samples']
    loodiffs = store['loodiffs']
    sample_diagnostics = store['sample_diagnostics']

resp_align = options.resp_align
delay = options.delay

labels = samples.index.levels[0]
area = labels[0][2:-7]
times = samples.index.levels[1]


#%% plot linear model betas and linear vs. step model comparison
fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(7.5, 8))

regressors = {'choice': 'response-signed', 'dotx': 'x-coordinate'}
regcols = {'choice': 'C0', 'dotx': 'C1'}

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
            for regressor in regressors.keys():
                ax.fill_between(
                        times, 
                        quantiles.loc[(slice(None), 0.025), 
                                      regressor].values.flatten(),
                        quantiles.loc[(slice(None), 0.975), 
                                      regressor].values.flatten(), 
                        facecolor=regcols[regressor], alpha=0.2)
                
            # plot medians
            for regressor in regressors.keys():
                ax.plot(times, quantiles.loc[(slice(None), 0.5), regressor], 
                        lw=2, color=regcols[regressor], 
                        label=regressors[regressor])
        
        elif var == 'loo':
            # plot 2*SE bands
            for regressor in regressors.keys():
                subset = loodiffs.loc[(label, slice(None)), regressor]
                ax.fill_between(
                        times, 
                        subset['diff'] - 2*subset['se'],
                        subset['diff'] + 2*subset['se'], 
                        facecolor=regcols[regressor], alpha=0.2)
                
            # plot loodiff
            for regressor in regressors.keys():
                subset = loodiffs.loc[(label, slice(None)), regressor]
                ax.plot(times, subset['diff'], lw=2,
                        color=regcols[regressor], label=regressors[regressor])
            
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
        area, delay)
fig.savefig(os.path.join(helpers.figdir, figname), dpi=300)


#%% check whether beta_lin or beta_step are larger across several time points
tsplit = -170
tfirst = -2000
tlast = 110

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 4))

ylabel = 'x-coordinate - response-signed (posterior mean beta)'

for label, ax in zip(labels, axes):
    subset = samples.loc[(label, slice(tfirst, tsplit), slice(None)),
                         (slice(None), 'beta_lin')]
    means = subset.groupby('time').mean()
    subset = samples.loc[(label, slice(tsplit+1, tlast), slice(None)),
                         (slice(None), 'beta_lin')]
    means = pd.concat([means, subset.groupby('time').mean()],
                      keys=['early', 'late'], names=['period', 'time'])
    means.columns = means.columns.levels[0].map(lambda x: regressors[x])
    
    diff = means[regressors['dotx']] - means[regressors['choice']]
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
                color=regcols['dotx'], zorder=0, alpha=0.2)
axes[0].fill_between(ax.get_xlim(), [yl[0], yl[0]], 
                color=regcols['choice'], zorder=0, alpha=0.2)
axes[1].fill_between(ax.get_xlim(), [yl[1], yl[1]], 
                color=regcols['choice'], zorder=0, alpha=0.2)
axes[1].fill_between(ax.get_xlim(), [yl[0], yl[0]], 
                color=regcols['dotx'], zorder=0, alpha=0.2)
axes[0].set_ylim(yl)

axes[1].set_ylabel('')
fig.tight_layout()

# svg didn't work with transparency in LibreOffice
figname = 'linearity_orig-flipped_%s_%s_%s_delay%d.png' % (
        r_name, 'response-aligned' if resp_align else 'firstdot-aligned',
        area, delay)
fig.savefig(os.path.join(helpers.figdir, figname), dpi=300)


#%% plot choice vs. dotx model comparison
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 5))

mcols = dict(linear='C2', step='C3')

for label, ax in zip(labels, axes):
    # plot 2*SE bands
    for model in mcols.keys():
        subset = loodiffs.loc[(label, slice(None)), model]
        ax.fill_between(
                times, 
                subset['diff'] - 2*subset['se'],
                subset['diff'] + 2*subset['se'], 
                facecolor=mcols[model], alpha=0.2)
        
    # plot loodiff
    for model in mcols.keys():
        subset = loodiffs.loc[(label, slice(None)), model]
        ax.plot(times, subset['diff'], lw=2,
                color=mcols[model], label=model)
    
    ax.plot(times[[0, -1]], [0, 0], '--k', lw=1, zorder=1)
    
    ax.set_title(label[0] + '-' + area)
    
    if resp_align:
        ax.set_xlabel('time from response (ms)')
    else:
        ax.set_xlabel('time from first dot onset (ms)')
        
axes[0].set_ylabel('evidence for stimulus encoding')

# plot response indicator
if resp_align and times[-1] >= 0:
    yl = axes[0].get_ylim()
    for ax in axes:
        ax.plot([0, 0], yl, ':k', label='time of response', zorder=1)
    ax.set_ylim(yl);

axes[0].legend(loc='lower left')

fig.tight_layout()


#%% plot evidence vs. response model comparison
fig = plt.figure(figsize=(8, 4))
# define two independent grids, because I want to easily position the first two
# axes together and the third separately
grid = GridSpec(1, 2, left=0.05, bottom=0.15, right=0.7, top=0.9)
bp_grid = GridSpec(1, 1, left=0.75, bottom=0.15, right=0.95, top=0.9)
axes = plt.subplot(grid[0, 0])
axes = [axes, plt.subplot(grid[0, 1], sharex=axes, sharey=axes),
        plt.subplot(bp_grid[0, 0])]

endt = 0
plottimes = list(times[times <= endt])
tsplit = -170

col= 'k'
evidcol = 'C0'
respcol = 'C1'

# plot model comparison across time
for label, ax in zip(labels, axes[:2]):
    subset = loodiffs.loc[(label, plottimes), 'ev_vs_resp']
    ax.fill_between(
            plottimes, 
            subset['diff'] - 2*subset['se'],
            subset['diff'] + 2*subset['se'], 
            facecolor=col, alpha=0.2)
        
    # plot loodiff
    subset = loodiffs.loc[(label, plottimes), 'ev_vs_resp']
    ax.plot(plottimes, subset['diff'], lw=2, color=col)
    
    ax.set_title(label[0] + '-' + area)
    
    if resp_align:
        ax.set_xlabel('time from response (ms)')
    else:
        ax.set_xlabel('time from first dot onset (ms)')
        
axes[0].set_ylabel('cross-validated log-likelihood difference')

# plot color box annotations and set axis limits
yl = [loodiffs.loc[(labels[0], plottimes), ('ev_vs_resp', 'diff')].min(), 
      0.004]
for ax in axes[:2]:
    ax.set_ylim(yl)
    ax.set_autoscale_on(False)
    
    ax.plot(ax.get_xlim(), [0, 0], '--k', lw=1, zorder=1)
    
    ax.fill_between(ax.get_xlim(), [yl[1], yl[1]], 
                    color=evidcol, zorder=0, alpha=0.2)
    ax.fill_between(ax.get_xlim(), [yl[0], yl[0]], 
                    color=respcol, zorder=0, alpha=0.2)
    
    ax.set_xlim(ax.get_xlim()[0], plottimes[-1])

axes[1].set_yticklabels([])

# plot response indicator
if resp_align and plottimes[-1] >= 0:
    for ax in axes[:2]:
        ax.plot([0, 0], yl, ':k', label='time of response', zorder=1)

axes[0].text(plottimes[0], 0.002, 'supports evidence', color=evidcol, fontsize=14)
axes[0].text(plottimes[0], -0.004, 'supports choice', color=respcol, fontsize=14)

# box plot for early and late time points
diff = loodiffs.loc[(slice(None), slice(plottimes[0], tsplit-1)), 
                    ('ev_vs_resp', 'diff')]
diff.name = 'diff'

print(label)
for label in labels:
    tval, pval = scipy.stats.ttest_1samp(diff[label], 0)
    print('%5s period: t = %+4.2f, p = %6.4f' % (label, tval, pval))

diff = diff.reset_index('label')
diff.label = diff.label.map(lambda s: s[0] + '-' + area)
sns.boxplot(x='label', y='diff', data=diff, ax=axes[2], color='0.5', 
            width=0.6, notch=True)
axes[2].set_ylabel('')
axes[2].set_xlabel('area')
axes[2].set_title('times < %d ms' % tsplit)

axes[2].set_autoscale_on(False)
xl = axes[2].get_xlim()
yl = axes[2].get_ylim()
axes[2].plot(xl, [0, 0], '--k', lw=1, zorder=0)
axes[2].fill_between(xl, [yl[1], yl[1]], color=evidcol, zorder=-1, alpha=0.2)
axes[2].fill_between(xl, [yl[0], yl[0]], color=respcol, zorder=-1, alpha=0.2)

# make tight layout
grid.tight_layout(fig, rect=[0, 0, 0.7, 0.97])
bp_grid.tight_layout(fig, rect=[0.68, 0, 1, 0.97])

fig.text(0.02, 0.92, 'A', weight='bold', fontsize=20)
fig.text(0.71, 0.92, 'B', weight='bold', fontsize=20)

fig.savefig(os.path.join(helpers.figdir, 'evidence_vs_response_%s.png' % area), 
            dpi=300)