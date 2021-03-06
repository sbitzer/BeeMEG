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
import source_statistics as ss

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
r_names = ['saccade_x', 'dot_x', 'dot_y', 'sum_dot_x']
r_name = r_names[0]

# dots to investigate
dots = np.arange(1, 6)


#%% load data
# load data for both hemispheres
loadlabels = ['{}_{}_ROI-{}h'.format(hemi, area, hemi.lower()) for hemi in ['L', 'R']]

subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

srcfile = os.path.join(ss.bem_dir, file)

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

Bx = 8


#%% make correlation figure for a single regressor at different times
r_name = 'saccade_x'

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])

colors = sns.cubehelix_palette(len(loadtimes), start=1.8, rot=0, light=.75, dark=.25)

xrange = reg[r_name].quantile([0.1, 0.9])
x_bins = np.linspace(*xrange, Bx+1)
x_bins = (x_bins[:-1] + x_bins[1:]) / 2

for hemi, ax in zip(['L', 'R'], axes):
    for time, color in zip(loadtimes, colors):
        df = pd.concat([data.loc[time], reg], axis=1)
        label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
        grid = sns.regplot(r_name, label, df, x_estimator=np.mean, 
                           x_bins=x_bins, ax=ax, 
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


#%% compare correlations of two different regressors in same plot
time = loadtimes[-1]
names = ['sum_dot_x', 'dot_x']
linelabels = ['accumulated', 'momentary']

colors = sns.cubehelix_palette(len(loadtimes), start=0.5, rot=1.6, light=.75, 
                               dark=.25)
colors = colors[slice(1, 4, 2)]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])

df = pd.concat([data.loc[time], reg[names]], axis=1)
#df = df.loc[27]

# normalise regressor values so that you see the whole range of both of them
#df[names] = df[names] / df[names].std()

# exclude most data points for which regressor values are similar
#diff = np.abs(df.dot_x - df.sum_dot_x)
#df = df[diff > diff.quantile(0.8)]

corrs = df.corr()

xrange = df[names].stack().quantile([0.01, 0.99])
xpred = np.linspace(*xrange, 200)[:, None]

for hemi, ax in zip(['L', 'R'], axes):
    print('processing hemisphere %s ...' % hemi, flush=True)
    label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
    
    # statistical test for equality of correlations
    ci_low, ci_high = ss.dependent_corr(
        corrs.loc[label, names[0]], 
        corrs.loc[label, names[1]], 
        corrs.loc[names[0], names[1]], df.shape[0], method='zou')
    tval, pval = ss.dependent_corr(
        corrs.loc[label, names[0]], 
        corrs.loc[label, names[1]], 
        corrs.loc[names[0], names[1]], df.shape[0], method='steiger')
    print('stats for correlation with %s - correlation with %s = 0:' 
          % (names[0], names[1]))
    print('   ci: [% 6.3f, % 6.3f]' % (ci_low, ci_high))
    print('   t = % 5.2f, p = %f\n' % (tval, pval))
    
    for name, linelabel, color in zip(names, linelabels, colors):
        gpm = ss.fitgp(df[name], df[label], np.linspace(*xrange, 15))
        
        ss.plot_gp_result(ax, xpred, gpm, color, linelabel)
    
    ax.set_title(hemi)
    ax.set_xlabel('x-coordinate (px)')
        
axes[0].legend(loc='upper left')
axes[0].set_ylabel('normalised mean currents in area %s' % area)

fig.subplots_adjust(top=.92, right=.96, wspace=.12)
fname = os.path.join(figdir, 'correlation_%s_%s_vs_%s_%d.png' % (
        area, names[0], names[1], time))
fig.savefig(fname, dpi=300)


#%% compare correlations for one regressor for all vs. only early time points
time = loadtimes[-1]
mototimes = time/1000 + (dots - 1) * 0.1

fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=[7.5, 4.5])

choices = pd.concat([regressors.subject_trial.response.loc[subjects]] * dots.size,
                    keys=dots, names=['dot', 'subject', 'trial'])
choices = choices.reorder_levels(['subject', 'trial', 'dot']).sort_index()
rts = pd.concat([regressors.subject_trial.RT.loc[subjects]] * dots.size,
                keys=dots, names=['dot', 'subject', 'trial'])
rts = rts.reorder_levels(['subject', 'trial', 'dot']).sort_index()

df = pd.concat([data.loc[time], reg], axis=1)
df['motoresp'] = regressors.motoresp_lin(mototimes).loc[subjects].values

r_names = np.array(['motoresp', 'dot_x', 'sum_dot_x'])

colors = sns.cubehelix_palette(len(loadtimes), start=1.8, rot=0, light=.75, 
                               dark=.25)
colors = colors[slice(1, 4, 2)]

# use only trials in which response was to contralateral side
select_contra = False

# regressor to plot data for
r_name = 'dot_x'

xrange = df[r_name].quantile([0.1, 0.9])

for hemi, ax in zip(['L', 'R'], axes):
    print('plotting hemisphere %s ...' % hemi, flush=True)
    label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
    
    for select_early, selabel, color in zip(
            [True, False], ['without fast responses', 'all data'], colors):
        rdata = df[[label, r_name]].copy()
        datatime = ((rdata.index.get_level_values('dot').values - 1) * 0.1 
                    + time / 1000)
        select = np.ones_like(rdata[r_name], dtype=bool)
        if hemi == 'L':
            if select_contra:
                select = select & (choices == 1.0)
#            rdata = rdata[rdata[r_name] >= 0]
        else:
#            rdata = rdata[rdata[r_name] <= 0]
            if select_contra:
                select = select & (choices == -1.0)

        if select_early:
            select = select & ((rts - datatime) > 0.5)
            
        rdata = rdata[select]
        
#        regplot = sns.regplot
        regplot = ss.gpregplot
        gpm = regplot(r_name, label, rdata, x_estimator=np.mean, x_bins=8, 
                      ax=ax, line_kws={'color': color, 'label': selabel},
                      scatter_kws={'color': color}, xrange=xrange)
        
    ax.set_xlabel('x-coordinate')
    ax.set_title(hemi)
    if hemi == 'L':
        ax.set_ylabel('normalised mean currents in area %s' % area)
    else:
        ax.set_ylabel('')
        ax.legend()
        
fig.subplots_adjust(top=.92, right=.96, wspace=.12)

contrastr = {True: '_contra', False: ''}
fname = os.path.join(figdir, 'correlation_all_vs_early_%s_%d%s.png' % (
        area, time, contrastr[select_contra]))
fig.savefig(fname, dpi=300)
        
        
#%%  plot motor preparation aligned to time of response
r_name = 'motoresp'

colors = sns.cubehelix_palette(2, start=1.3, rot=0, light=.6, dark=.4)

xrange = regressors.linprepsig(np.r_[-1.15, 0.15])

figm, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])


def get_trials_xvals(choice):
    if choice == 'L':
        # select trials which had a left choice
        select = choices == -1.0
        
        # for left choices the motoresp_lin regressor is negative
        Z = np.linspace(*-xrange, 15)
        xpred = -np.linspace(*xrange, 200)[:, None]
        rttime = (-xpred - 1) * regressors.maxrt
    else:
        # select trials which had a right choice
        select = choices == 1.0
        
        # for right choices the motoresp_lin regressor is positive
        Z = np.linspace(*xrange, 15)
        xpred = np.linspace(*xrange, 200)[:, None]
        rttime = (xpred - 1) * regressors.maxrt
    
    return select, Z, xpred, rttime


hemis = ['L', 'R']
for hemi, ax in zip(hemis, axes):
    print('plotting hemisphere %s ...' % hemi, flush=True)
    label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
    
    for choice, color in zip(['contralateral', 'ipsilateral'], colors):
        rdata = df[[label, r_name]].copy()
        
        if choice == 'ipsilateral':
            plot_choice = hemi
        else:
            plot_choice = set(hemis).difference(hemi).pop()
        select, Z, xpred, rttime = get_trials_xvals(plot_choice)
        
        rdata = rdata[select]
        
        gpm = ss.fitgp(rdata[r_name], rdata[label], Z)
        
        ss.plot_gp_result(ax, xpred, gpm, color, choice, xx=rttime)

    ax.set_xlabel('time from response (s)')
    ax.set_title(hemi)
    
    if hemi == hemis[0]:
        yl = ax.get_ylim()
    ax.plot([0, 0], yl, ':k', label='time of response')
    

axes[0].set_ylabel('normalised mean currents in area %s' % area)
axes[0].legend()
ax.set_ylim(yl)

figm.subplots_adjust(top=.92, right=.96, wspace=.12)

fname = os.path.join(figdir, 'motoprep_signal_%s_%d.png' % (
        area, time))
figm.savefig(fname, dpi=300)
