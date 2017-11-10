#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:57:38 2017

@author: bitzer
"""

import numpy as np
import helpers
import regressors
import source_visualisations as sv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import GPy

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% select data
# source data file to use
# label mode = mean
file = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'

# area that should be investigated
area = '4'

# can be either 'gp' for sparse GP or 'mean' for simple average
method = 'gp'

timeslice = [0, 890]

normsrc = True


labels = ['{}_{}_ROI-{}h'.format(hemi, area, hemi.lower()) 
          for hemi in ['L', 'R']]

subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

srcfile = os.path.join(sv.bem_dir, file)

# get mean and std of data across all subjects, trials and times
if normsrc:
    epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
    epochs_std = pd.read_hdf(srcfile, 'epochs_std')


#%% helper for fitting sparse GP
def fitgp(xvals, data, Z):
    if data.ndim < 2:
        data = data[:, None]
    if xvals.ndim < 2:
        xvals = xvals[:, None]
    if Z.ndim < 2:
        Z = Z[:, None]
    
    gpm = GPy.models.SparseGPRegression(xvals, data, Z=Z)
    
    # set reasonable initial values
    gpm.likelihood.variance = data.var()
    gpm.kern.lengthscale = xvals.std()
    
    gpm.optimize()
    
    return gpm


#%% infer for each subject individually
choices = regressors.subject_trial.response.loc[subjects]

store = pd.HDFStore(srcfile, 'r')

Z = np.linspace(*timeslice, 15)
if method == 'gp':
    xpred = np.linspace(*timeslice, 200)
else:
    xpred = np.arange(timeslice[0], timeslice[1] + 10, 10)

pred = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [subjects, labels, xpred], 
                names=['subject', 'label', 'time']),
        columns=['mean', 'std'], dtype=float)

for sub in subjects:
    print('\rsubject = %2d' % sub, end='', flush=True)
    epochs = store.select('label_tc', 'subject=sub')
    
    times = epochs.index.levels[2]
    times = times[slice(*times.slice_locs(*timeslice))]
    
    for hemi in ['L', 'R']:
        label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
        
        # select trials of the contralateral choice
        if hemi == 'L':
            trials = choices.loc[sub] == 1
        else:
            trials = choices.loc[sub] == -1
        trials = list(trials[trials].index)
        
        data = epochs.loc[(sub, trials, slice(None)), label]
        
        # normalise data
        if normsrc:
            data = (data - epochs_mean[label]) / epochs_std[label]
        
        if method == 'gp':
            # fit the sparse GP
            gpm = fitgp(data.index.get_level_values('time').values, 
                        data.values, Z)
            
            # get the noiseless prediction
            gpmean, gpvar = gpm.predict_noiseless(xpred[:, None])
            pred.loc[(sub, label, slice(None)), 'mean'] = gpmean[:, 0]
            pred.loc[(sub, label, slice(None)), 'std'] = np.sqrt(gpvar[:, 0])
        else:
            pred.loc[(sub, label, slice(None)), 'mean'] = \
                data.groupby('time').mean().values
            pred.loc[(sub, label, slice(None)), 'std'] = \
                data.groupby('time').std().values / np.sqrt(subjects.size)
        
    gc.collect()
        

#%% plot
colors = sns.cubehelix_palette(2, start=0.9, rot=0, light=.6, dark=.2)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])

for ax, label in zip(axes, labels):
    labelpred = pred.xs(label, level='label').unstack('subject')
    lsubs = ax.plot(xpred / 1000, labelpred['mean'], color=colors[0], lw=1)
    
    # compute mean prediction across subjects taking uncertainty into account
    meanpred = labelpred['mean'].mean(axis=1)
    stdpred = np.sqrt((labelpred['std'] ** 2).sum(axis=1) / subjects.size ** 2)
#    stdpred = labelpred['mean'].std(axis=1) / np.sqrt(subjects.size)
    
    lmean, = ax.plot(xpred / 1000, meanpred, color=colors[1], lw=2)
    ax.fill_between(xpred / 1000, meanpred - 2 * stdpred, 
                    meanpred + 2 * stdpred, facecolor=colors[1], alpha=.15)
    
    ax.set_title(label[0])
    ax.set_xlabel('time from first dot onset (s)')

ax.legend([lsubs[0], lmean], ['participants', 'mean'])
axes[0].set_ylabel('normalised mean currents in area %s' % area)
fig.subplots_adjust(top=0.92, right=0.97, left=0.1, wspace=0.08)

fname = os.path.join(figdir, 'general_contralateral_buildup_%s_%s.png' % (
        area, method))
fig.savefig(fname, dpi=300)