#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:35:19 2017

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regressors
import os
import gc

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% select data
# source data file to use
# label mode = mean
file = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'

# area that should be investigated
area = '4'

# times after dot onset (in ms) to investigate
loadtimes = [120, 170, 320, 400]

# regressor name to investigate
# the first is the main one to investigate, following ones are used for comparison
r_names = ['dot_x', 'dot_y', 'sum_dot_x']
r_name = r_names[0]

# dots to investigate
dots = np.arange(1, 6)


#%% load data
# load data for both hemispheres
loadlabels = ['{}_{}_ROI-{}h'.format(hemi, area, hemi.lower()) for hemi in ['L', 'R']]

subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', file)

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

times = epochs.index.levels[2]

# index of last time point available for this analysis
tendi = times.searchsorted(times[-1] - (dots.max() - 1) * helpers.dotdt * 1000)

epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')

def extract_time_data(epochs, time):
    # get the times associated with the dots
    datat = time + (dots - 1) * helpers.dotdt * 1000
    assert np.all(datat <= times[-1]), ("The times where you are looking for an "
                  "effect have to be within the specified time window")
    datat = times[[np.abs(times - t).argmin() for t in datat]]

    data = (  epochs.loc[(slice(None), datat), :] 
            - epochs_mean[loadlabels]) / epochs_std[loadlabels]
    
    data.index = pd.MultiIndex.from_product([epochs.index.levels[0], dots], 
                                            names=['trial', 'dot'])
    
    return data

def load_normalised_data(sub):
    with pd.HDFStore(srcfile, 'r') as store:
        epochs = store.select('label_tc', 'subject=sub')
    
    return pd.concat([extract_time_data(epochs.loc[sub, loadlabels], time) 
                      for time in loadtimes],
                      keys=loadtimes, names=['time', 'trial', 'dot'])

data = pd.concat([load_normalised_data(sub) for sub in subjects],
                 keys=subjects, names=['subject', 'time', 'trial', 'dot'])

data = data.reorder_levels(['time', 'subject', 'trial', 'dot'])
data.sort_index(inplace=True)

gc.collect()

print('loaded data for area "{}"'.format(area))


#%% load regressors
reg = pd.concat([regressors.trial_dot[r_name].loc[(slice(None), list(dots))]] * subjects.size,
                keys=subjects, names=['subject', 'trial', 'dot'])
reg = pd.DataFrame(reg)
for name in r_names[1:]:
    reg[name] = np.tile(regressors.trial_dot[name].loc[(slice(None), list(dots))].values, subjects.size)


#%% make correlation figure for a single regressor at different times
r_name = 'dot_x'

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])

colors = sns.cubehelix_palette(len(loadtimes), start=2, rot=.1, light=.75, dark=.25)

for hemi, ax in zip(['L', 'R'], axes):
    for time, color in zip(loadtimes, colors):
        df = pd.concat([data.loc[time], reg], axis=1)
        label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
        grid = sns.regplot(r_name, label, df, x_estimator=np.mean, 
                           x_bins=7, ax=ax, 
                           line_kws={'color': color, 'label': '%d ms' % time},
                           scatter_kws={'color': color})
    
    ax.set_title(hemi)
    ax.set_xlabel('x-coordinate')
    if hemi=='L':
        ax.set_ylabel('normalised mean currents in area %s' % area)
    else:
        ax.set_ylabel('')
        ax.legend()
        
fig.subplots_adjust(top=.92, right=.96, wspace=.12)
fig.savefig(os.path.join(figdir, 'correlation_%s_%s.png' % (area, r_name)), 
            dpi=300)


#%% compare correlations of regressors at one time point
time = loadtimes[-1]
mototimes = time/1000 + (dots - 1) * 0.1

colors = sns.cubehelix_palette(3, start=0, rot=0.8, light=0.8, dark=0.2)

fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=[7.5, 4.5])

df = pd.concat([data.loc[time], reg], axis=1)
df['motoresp'] = regressors.motoresp_lin(mototimes).loc[subjects].values

leglabels = ['x-coordinate', 'sum of x', 'linear rise']

for hemi, ax in zip(['L', 'R'], axes):
    label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
    
    for r_name, color, leglabel in zip(['dot_x', 'sum_dot_x', 'motoresp'], 
                                       colors, leglabels):
        rdata = df[[label, r_name]]
        if hemi == 'L':
            rdata = rdata[rdata[r_name] >= 0]
        else:
            rdata = rdata[rdata[r_name] <= 0]
        rdata[r_name] = (rdata[r_name] - rdata[r_name].mean()) / rdata[r_name].std()
        
        ax = sns.regplot(r_name, label, rdata, x_estimator=np.mean, 
                           x_bins=7, ax=ax, 
                           line_kws={'color': color, 'label': leglabel},
                           scatter_kws={'color': color})
    
    ax.set_xlabel('normalised regressor value')
    ax.set_title(hemi)
    if hemi == 'L':
        ax.set_ylabel('normalised mean currents in area %s' % area)
    else:
        ax.set_ylabel('')
        ax.legend()
        
fig.subplots_adjust(top=.92, right=.96, wspace=.12)
fig.savefig(os.path.join(figdir, 'correlation_comparison_%s_%d.png' % (area, time)), 
            dpi=300)