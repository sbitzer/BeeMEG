#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:30:26 2017

@author: bitzer
"""

import numpy as np
import helpers
import regressors
import subject_DM
import source_statistics as ss
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_1samp
import statsmodels.api as sm
import pystan
import gc


#%% options

# label mode = mean
srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201712061743.h5'

srcfile = os.path.join(ss.bem_dir, srcfile)

area = 'v23ab'

# set to 'local' to normalise within time points and subjects
# set to 'global' to normalise across the complete design matrix
# set to anything else to not normalise
# applies to both data and design matrix
normalise = 'local'

labels = ['L_%s_ROI-lh' % area, 'R_%s_ROI-rh' % area]

delay = 400

resp_align = True

# allows you to exclude trials based on response-aligned time in first dot 
# onset aligned analysis: all trials with response-aligned time >= toolate will
# be removed from regression analysis, set to 5000 or higher to include all 
# available trials
toolate = 5000

if resp_align:
    timeslice = [-1000, delay]
    fbase = 'response-aligned'
else:
    timeslice = [delay, 1500]
    fbase = 'firstdot-aligned'

fbase = '%s_%s_norm%s_delay%3d_toolate%+05d' % (
            fbase, area, normalise, delay, toolate)
    
times = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                 name='time')

r_colors = {'intercept': 'C0', 'dot_x_time': 'C1', 'dot_y_time': 'C2', 
            'percupt_x_time': 'C3', 'percupt_y_time': 'C4'}
r_labels = {'percupt_y_time': 'PU-y', 
            'percupt_x_time': 'PU-x',
            'dot_x_time': 'x-coord',
            'dot_y_time': 'y-coord',
            'intercept': 'intercept'}
r_names = list(r_labels.keys())


#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')
epochtimes = epochs.index.levels[2]


#%% define the function which reindexes time to response-aligned
choices = regressors.subject_trial.response.loc[subjects]
rts = regressors.subject_trial.RT.loc[subjects] * 1000

# if RTs are an exact multiple of 5, numpy will round both 10+5 and 20+5 to
# 20; so to not get duplicate times add a small jitter
fives = (np.mod(rts, 10) != 0) & (np.mod(rts, 5) == 0)
# subtracting the small amount means that the new times will be rounded
# down so that e.g., -5 becomes 0, I prefer that round because then there
# might be slightly more trials with larger times, but the effect is small
rts[fives] = rts[fives] - 1e-7

def rtindex(sub):
    """Returns index for data of a subject with time as difference to RT"""
    
    rttime = (  epochtimes.values[None, :] 
              - rts.loc[sub].values[:, None]).flatten()
    # round to 10 ms and convert into integer
    rttime = rttime.round(-1).astype(int)

    # create new subject-trial-time index (I cannot only set the labels of
    # the time level, because the response aligned times have different 
    # values than those stored in dot-onset aligned times)
    return pd.MultiIndex.from_arrays(
            [np.tile(np.arange(1, 481)[:, None], 
                     (1, epochtimes.size)).flatten(),
             rttime], names=['trial', 'time']).sort_values()


#%% load data of all subjects
def load_subject(sub):
    print('\rloading subject %02d ...' % sub, flush=True, end='')
    with pd.HDFStore(srcfile, 'r') as store:
        epochs = store.select('label_tc', 'subject=sub')[labels]
    
    if resp_align:
        # get response-aligned times of data points
        epochs.index = rtindex(sub)
        
        # exclude trials which were timed out (were coded as RT=5000 ms)
        epochs = epochs.loc[(rts.loc[sub][rts.loc[sub] <= 2500].index.values, 
                             slice(None))]
    else:
        epochs = epochs.loc[sub]
        
    gc.collect()
    
    return epochs

print("loading ... ", end='')
alldata = pd.concat([load_subject(sub) for sub in subjects],
                    keys=subjects, names=['subject', 'trial', 'time'])
print('done.')

if normalise == 'global':
    epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
    epochs_std = pd.read_hdf(srcfile, 'epochs_std')
    
    alldata = (alldata - epochs_mean[labels]) / epochs_std[labels]

    
#%% create response-aligned design matrix
DM = subject_DM.get_trial_time_DM(r_names, epochtimes.values / 1000, 
                                  delay=delay / 1000, subjects=subjects)

DM.sort_index(axis=1, inplace=True)

if normalise == 'global':
    DM = subject_DM.normalise_DM(DM, True)

if resp_align:
    # exclude trials in which there was no response (timed out trials)
    def get_response_trials(sub):
        DMsub = DM.loc[sub]
        rtssub = rts.loc[sub]
        return DMsub.loc[(rtssub[rtssub != helpers.toresponse[1] * 1000].index, 
                          slice(None)), :]
    
    DM = pd.concat([get_response_trials(sub) for sub in subjects],
                    keys=subjects, names=['subject', 'trial', 'time'])
    
# now data and regressor are in the same order, so simply assign the 
# response-aligned index of alldata to the regressor values to get 
# response-aligned regressor values, if not response-aligned this transforms
# the time index from seconds to milliseconds
DM.index = alldata.index
    

#%% standard summary statistics inference
first_level = pd.DataFrame([],
        index=pd.MultiIndex.from_product(
                [labels, times], names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [subjects, ['beta', 'bse'], DM.columns], 
                names=['subject', 'measure', 'regressor']), dtype=float)
first_level.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, times], names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [['mean', 'std', 'tval', 'mlog10p'], DM.columns], 
                names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

for label in labels:
    trial_counts = []
    
    for time in times:
        print('\rlabel %s, time = % 5d' % (label, time), end='', flush=True)
        
        data = pd.concat([DM.xs(time, level='time'), 
                          alldata.xs(time, level='time')], axis=1)
        
        # remove trials which are > -100 to response
        if not resp_align:
            data = data.loc[rts[(time - rts) < toolate].index]
        
        data = data.dropna()
        
        # count remaining trials
        trial_counts.append(data[label].groupby('subject').count())
        
        for sub in subjects:
            datasub = data.loc[sub].copy()
            
            if datasub.shape[0] > 30:
                if normalise == 'local':
                    # subtracts the mean when its magnitude is more than 
                    # twice the std (see msratio)
                    datasub[DM.columns] = subject_DM.normalise_DM(
                            datasub[DM.columns], True)
                    
                    # only normalise std (mean to be capture by intercept)
                    datasub[label] /= datasub[label].std()
                
                res = sm.OLS(
                        datasub[label].values, 
                        datasub[first_level.columns.levels[2]].values, 
                        hasconst=True).fit()
                first_level.loc[(label, time), (sub, 'beta', slice(None))] = (
                        res.params)
                first_level.loc[(label, time), (sub, 'bse', slice(None))] = (
                        res.bse)
        
        params = first_level.loc[(label, time), 
                                 (slice(None), 'beta', slice(None))]
        params = params.unstack('regressor')
        
        params.dropna(inplace=True)
        
        second_level.loc[(label, time), ('mean', slice(None))] = (
                params.mean().values)
        second_level.loc[(label, time), ('std', slice(None))] = (
                params.std().values)
        tvals, pvals = ttest_1samp(params.values, 0, axis=0)
        second_level.loc[(label, time), ('tval', slice(None))] = tvals
        second_level.loc[(label, time), ('mlog10p', slice(None))] = (
                -np.log10(pvals))

print('')

trial_counts = pd.concat(trial_counts, axis=1, keys=times).stack()


#%% plot number of data points per subject for plotted times
sl = second_level.dropna()
tmin = sl.index.get_level_values('time').min()
tmax = sl.index.get_level_values('time').max()

fig, ax = plt.subplots(1, figsize=(7.5, 4))

cnts = trial_counts.loc[(slice(None), slice(tmin, tmax))].unstack('subject')

lines = ax.plot(cnts.index, cnts.values, color='k', alpha=0.5)

if resp_align:
    ax.set_xlabel('time from response (ms)')
else:
    ax.set_xlabel('time from first dot onset (ms)')
ax.set_ylabel('trial count')
ax.legend([lines[0]], ['single subjects'], loc='center right')

fig.savefig(os.path.join(helpers.figdir, 'trcounts_%s.png' % fbase), dpi=300)


#%% plot results
show_measure = 'mean'
alpha = 0.01

sl = second_level.dropna().copy()
ss.add_measure(sl, 'mlog10p_fdr')

r_name = 'dot_x_time'

# which regressors to show? 
# add main regressor of interest to end to show it on top
show_names = ['percupt_x_time', 'dot_y_time', 'percupt_y_time']
show_names.append(r_name)

fig, axes = plt.subplots(1, len(labels), sharey=True, figsize=(7.5, 4))

for label, ax in zip(labels, axes):
    lines = {}
    for rn in show_names:
        lines[rn], = ax.plot(sl.loc[label].index, 
                                 sl.loc[label].loc[:, (show_measure, rn)], 
                                 label=r_labels[rn], color=r_colors[rn])
    
    ti = sl.loc[label].index
    
    # determine whether positive or negative peaks are larger
    mm = pd.Series([sl.loc[label].loc[:, (show_measure, r_name)].min(),
                    sl.loc[label].loc[:, (show_measure, r_name)].max()])
    mm = mm[mm.abs().idxmax()]
    
    # plot markers over those time points which are significant after FDR
    significant = np.full(ti.size, mm * 1.1)
    significant[sl.loc[label, ('mlog10p_fdr', r_name)] 
                < -np.log10(alpha)] = np.nan
    ax.plot(ti, significant, '*', label='significant',
            color=lines[r_name].get_color())
    ax.plot(ti[[0, -1]], [0, 0], '--k', zorder=1)
    
    ax.set_title(label[0] + '-' + area)
    if resp_align:
        ax.set_xlabel('time from response (ms)')
    else:
        ax.set_xlabel('time from first dot onset (ms)')

if resp_align:
    yl = ax.get_ylim()
    for ax in axes:
        ax.plot([0, 0], yl, ':k', label='response', zorder=1)
    ax.set_ylim(yl)

ax.legend()

axes[0].set_ylabel('estimated second-level beta')

fig.tight_layout()
fig.savefig(os.path.join(helpers.figdir, 
                         '%s_%s.png' % (r_name[:-5], fbase)), dpi=300)


#%% estimate second-level time course with GPs
xrange = timeslice
bin_width = 100
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 5))

fl = (first_level.xs('beta', level='measure', axis=1)
      .stack('subject').reset_index('time'))
fl = fl[(fl.time >= xrange[0]) & (fl.time <= xrange[1])]

r_name = 'dot_x_time'
# which regressors to show? 
# add main regressor of interest to end to show it on top
show_names = ['percupt_x_time', 'dot_y_time']
show_names.append(r_name)

for label, ax in zip(labels, axes):
    for rn in show_names:
        color = r_colors[rn]
        r_label = r_labels[rn]
        
        gpm = ss.gpregplot(
                'time', rn, fl.loc[label], x_estimator=np.mean, 
                x_bins=int((xrange[1] - xrange[0]) / bin_width), 
                line_kws={'color': color, 'label': r_label},
                scatter_kws={'color': color}, xrange=xrange, ax=ax,
                gp_kws={'gptype': 'heteroscedastic', 'smooth': False})
    
    ax.plot(xrange, [0, 0], '--k', zorder=1)
    ax.set_ylabel('')
    ax.set_title(label[0] + '-' + area)
    if resp_align:
        ax.set_xlabel('time from response (ms)')
    else:
        ax.set_xlabel('time from first dot onset (ms)')
    
if resp_align and xrange[1] >= 0:
    yl = axes[0].get_ylim()
    for ax in axes:
        ax.plot([0, 0], yl, ':k', label='response', zorder=1)
    axes[0].set_ylim(yl);

axes[0].set_ylabel('estimated second-level beta')
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(helpers.figdir, 
                         '%s_%s_gp.png' % (r_name[:-5], fbase)), dpi=300)


#%% check relationship at particular time point before response
r_name = 'dot_x_time'
time = 100

data = pd.concat([DM.xs(time, level='time')[r_name], 
                  alldata.xs(time, level='time')], axis=1)
data = data.dropna()

# flip x-values according to the eventual choice; this allows you to check 
# whether a linear relationship only results because there are only two levels
# of activity (one for left and one for right choice), but these are mixed with
# a probability (of choosing left or right) given by the x-value; if this is 
# the case you should see the two levels of activity after flipping instead of
# a linear relationship
data[r_name] *= np.sign(data[r_name]) * np.sign(choices.loc[data.index])

color = r_colors[r_name]
xrange = data[r_name].quantile([0.1, 0.9])
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 4.5))
for i in range(2):
    gpm = ss.gpregplot(r_name, labels[i], data, x_estimator=np.mean, x_bins=8,
                       line_kws={'color': color, 'label': '%d ms' % time},
                       scatter_kws={'color': color}, xrange=xrange, ax=axes[i])


#%% create Stan model
#hierarchical_stan = """
#data {
#    int<lower=1> S;
#    int<lower=0> N[S];
#    int<lower=1> Nall;
#     
#    vector[Nall] reg;
#    vector[Nall] y;
#}
#
#parameters {
#    real z;
#    real<lower=0> stdx;
#    
#    vector[S] x;
#    vector[S] intercept;
#    
#    vector<lower=0>[S] stdy;
#}
#
#transformed parameters {
#    vector[S] diffs = x * stdx;
#    vector[S] meany = diffs + z;
#}
#
#model {
#    int n = 0;
#    
#    z ~ normal(0, 1);
#    x ~ normal(0, 1);
#    intercept ~ normal(0, 1);
#    stdx ~ exponential(1);
#    stdy ~ exponential(1);
#    
#    for (sub in 1:S) {
#        y[n+1 : n+N[sub]] ~ normal(reg[n+1 : n+N[sub]] * meany[sub]
#                                   + intercept[sub], stdy[sub]);
#        
#        n = n + N[sub];
#    }
#}
#"""
#
#def get_stan_data(time, label):
#    data = pd.concat([sumx_normalised.xs(time, level='time'), 
#                      alldata.xs(time, level='time')[label]],
#             axis=1)
#    data = data.dropna()
#    N = data.sum_dot_x.groupby('subject').count()
#    
#    return {'S': N.size,
#            'N': N.values,
#            'Nall': N.sum(),
#            'reg': data.sum_dot_x.values,
#            'y': data[label].values}
#
## pre-compile
#fit = pystan.stan(model_code=hierarchical_stan, 
#                  data=get_stan_data(time, label),
#                  iter=1000, chains=4)
#
#
##%%
#fit2 = pystan.stan(fit=fit, data=get_stan_data(time, label),
#                   iter=1000, chains=4)