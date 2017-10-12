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
import GPy

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


#%% plot helper
def plot_gp_result(ax, xpred, gpm, color, label, xx=None):
    if xx is None:
        xx = xpred
        
    gpmean, gpvar = gpm.predict_noiseless(xpred)
    
    gpmean = gpmean[:, 0]
    gpstd = np.sqrt(gpvar[:, 0])
    
    lw = plt.rcParams["lines.linewidth"] * 1.5
    
    # Draw the regression line and confidence interval
    ax.plot(xx, gpmean, color=color, lw=lw, label=label)
    ax.fill_between(xx[:, 0], gpmean - 2 * gpstd, gpmean + 2 * gpstd, 
                    facecolor=color, alpha=.15)
    

def gpregplot(r_name, label, rdata, x_estimator=None, x_bins=10, ax=None, 
              line_kws=None, scatter_kws=None, xrange=[-1.5, 1.5]):
    gpm = gpm = GPy.models.SparseGPRegression(
            rdata[r_name].values[:, None], 
            rdata[label].values[:, None], 
            Z=np.linspace(-1.5, 1.5, 15)[:, None])
    gpm.optimize()
    
    xpred = np.linspace(*xrange, 200)[:, None]
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # plot regression function
    plot_gp_result(ax, xpred, gpm, line_kws['color'], line_kws['label'])
    
    # scatterplot
    bin_edges = np.linspace(*xrange, x_bins+1)
    binx = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax = sns.regplot(r_name, label, rdata, x_estimator=np.mean, x_bins=binx, 
                     ax=ax, scatter_kws={'color': color}, fit_reg=False)
    
    return gpm


#%% compare correlations of regressors at one time point and 
#  plot motor preparation aligned to time of response
time = loadtimes[-1]
mototimes = time/1000 + (dots - 1) * 0.1

colors = sns.cubehelix_palette(3, start=0, rot=0.8, light=0.8, dark=0.2)

fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=[7.5, 4.5])
figm, axm = plt.subplots(1, figsize=[5, 4.5])

choices = pd.concat([regressors.subject_trial.response.loc[subjects]] * dots.size,
                    keys=dots, names=['dot', 'subject', 'trial'])
choices = choices.reorder_levels(['subject', 'trial', 'dot']).sort_index()
rts = pd.concat([regressors.subject_trial.RT.loc[subjects]] * dots.size,
                keys=dots, names=['dot', 'subject', 'trial'])
rts = rts.reorder_levels(['subject', 'trial', 'dot']).sort_index()

df = pd.concat([data.loc[time], reg], axis=1)
df['motoresp'] = regressors.motoresp_lin(mototimes).loc[subjects].values

leglabels = ['motor' ,'momentary', 'accumulated']

select_contra = True
select_early = False

for hemi, ax in zip(['L', 'R'], axes):
    print('plotting hemisphere %s ...' % hemi, flush=True)
    label = '{}_{}_ROI-{}h'.format(hemi, area, hemi.lower())
    
    for r_name, color, leglabel in zip(['motoresp', 'dot_x', 'sum_dot_x'], 
                                       colors, leglabels):
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
        rmean = rdata[r_name].mean()
        rstd = rdata[r_name].std()
        rdata[r_name] = (rdata[r_name] - rmean) / rstd
        
#        regplot = sns.regplot
        regplot = gpregplot
        gpm = regplot(r_name, label, rdata, x_estimator=np.mean, x_bins=8, 
                      ax=ax, line_kws={'color': color, 'label': leglabel},
                      scatter_kws={'color': color})
        
        if r_name == 'motoresp':
            if hemi == 'L':
                xpred = np.linspace(-1.5, 1.5, 200)[:, None]
                rttime = (xpred * rstd + rmean - 1) * regressors.maxrt
                gpcol = color
            else:
                xpred = np.linspace(1.5, -1.5, 200)[:, None]
                rttime = (-(xpred * rstd + rmean) - 1) * regressors.maxrt
                # make color a bit darker
                gpcol = 0.7 * np.array(gpcol)
            
            plot_gp_result(axm, xpred, gpm, gpcol, hemi, xx=rttime)
            
    
    ax.set_xlabel('normalised regressor value')
    ax.set_title(hemi)
    if hemi == 'L':
        ax.set_ylabel('normalised mean currents in area %s' % area)
    else:
        ax.set_ylabel('')
        ax.legend()

axm.set_xlabel('time from response (s)')
axm.set_ylabel('normalised mean currents in area %s' % area)
axm.plot([0, 0], axm.get_ylim(), ':k', label='time of response')
axm.legend()

figm.subplots_adjust(top=0.95, right=0.95, left=0.15)

contrastr = {True: '_contra', False: ''}
earlystr = {True: '_early', False: ''}
fname = os.path.join(figdir, 'motoprep_signal_%s_%d%s%s.png' % (
        area, time, contrastr[select_contra], earlystr[select_early]))
figm.savefig(fname, dpi=300)

fig.subplots_adjust(top=.92, right=.96, wspace=.12)
fname = os.path.join(figdir, 'correlation_comparison_%s_%d%s%s.png' % (
        area, time, contrastr[select_contra], earlystr[select_early]))
fig.savefig(fname, dpi=300)

