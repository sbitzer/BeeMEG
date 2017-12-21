#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:45:31 2017

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

linstepfun = lambda x, beta, l: beta * 2 * (1 / (1 + np.exp(-(x / l))) - 0.5)


#%% load results (samples)
# resp-align, delay=400, times=[-700, 400], area 4
fname = 'source_linearity_201712181904.h5'
with pd.HDFStore(os.path.join(helpers.resultsdir, fname), 'r') as store:
    samples = store['samples']
    r_name = store['r_name'][0]
    options = store['scalar_params']

resp_align = options.resp_align

labels = samples.index.levels[0]
area = labels[0][2:-7]
times = samples.index.levels[1]


#%% plot some results
fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(7.5, 8))

fliplabels = {True: 'flipped', False: 'original'}
flipcols = {True: 'C0', False: 'C1'}

varlabels = {'beta': r'population-level effect ($\beta$)', 
             'l': 'linearity (l)'}

for var, row in zip(samples.columns.levels[1], axes):
    for label, ax in zip(labels, row):
        subset = samples.loc[(label, slice(None)), (slice(None), var)]
        
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
            ax.plot([0, 0], yl, ':k', label='response', zorder=1)
        row[0].set_ylim(yl);

axes[0, 0].set_ylim(-0.4, 0.4)

axes[0, 0].legend(loc='lower left')

fig.tight_layout()

figname = 'linearity_%s_%s_%s_delay%d.svg' % (
        r_name, 'response-aligned' if resp_align else 'firstdot-aligned',
        area, options.delay)
fig.savefig(os.path.join(helpers.figdir, figname), dpi=300)


#%% plot estimated functional relationship
time = -100
flip = True

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 4))

xx = np.linspace(-3, 3, 500)
for label, ax in zip(labels, axes):
    medians = samples.loc[(label, time, slice(None)), flip].median()
    
    ax.plot(xx, linstepfun(xx, medians['beta'], medians['l']))
    
    ax.set_title('%s-%s (%d ms)' % (label[0], area, time))
    ax.set_xlabel('normalised %s value' % r_name)

axes[0].set_ylabel('population level signal (fraction of std)')

fig.tight_layout()


#%% demonstrate linear-step-regression-function
beta = 0.04
intercept = 0
ls = np.r_[0.01, 0.3, 1.3]

xx = np.linspace(-3, 3, 500)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
colors = sns.cubehelix_palette(ls.size, 0.8, light=0.7)

# plot the curves
for l, col in zip(ls, colors):
    ax.plot(xx, linstepfun(xx, beta, l) + intercept, label='l = %4.2f' % l,
            color=col)

# add annotations
yl = ax.get_ylim()
xl = ax.get_xlim()
ax.plot(xl, [-beta, -beta], '--k', lw=1, zorder=1)
ax.plot(xl, [beta, beta], '--k', lw=1, zorder=1)
ax.set_xlim(xl)

ax.text(1.1 * xl[1], beta, r'$\beta$', 
        verticalalignment='center', horizontalalignment='right')
ax.text(1.1 * xl[1], -beta, r'-$\beta$', 
        verticalalignment='center', horizontalalignment='right')

ax.set_ylabel('signal')
ax.set_xlabel('regressor value')
ax.legend()

fig.subplots_adjust(0.18, 0.15, 0.93, 0.95)
fig.savefig(os.path.join(helpers.figdir, 'linstepfun.svg'), dpi=300)