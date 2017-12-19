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


#%% load results (samples)
# resp-align, delay=400, times=[-700, 400], area 4
fname = 'source_linearity_201712181904.h5'
with pd.HDFStore(os.path.join(helpers.resultsdir, fname), 'r') as store:
#    samples = store['samples']
    r_name = store['r_name'][0]
    options = store['scalar_params']

resp_align = options.resp_align
    
samples = pd.read_hdf(os.path.join(helpers.resultsdir, fname + '.tmp'),
                      'samples')

labels = samples.index.levels[0]
area = labels[0][2:-7]
times = samples.index.levels[1]


#%% plot some results
fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(7.5, 8))

fliplabels = {True: 'flipped', False: 'original'}
flipcols = {True: 'C0', False: 'C1'}

varlabels = {'beta': 'population-level effect', 'l': 'linearity'}

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

axes[0, 0].legend(loc='upper left')

fig.tight_layout()


#%% plot estimated functional relationship
time = -100
flip = True

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 4))

fun = lambda x, beta, l: beta * 2 * (1 / (1 + np.exp(-(x / l))) - 0.5)

xx = np.linspace(-3, 3, 500)
for label, ax in zip(labels, axes):
    medians = samples.loc[(label, time, slice(None)), flip].median()
    
    ax.plot(xx, fun(xx, medians['beta'], medians['l']))
    
    ax.set_title('%s-%s (%d ms)' % (label[0], area, time))
    ax.set_xlabel('normalised %s value' % r_name)

axes[0].set_ylabel('population level signal (fraction of std)')

fig.tight_layout()