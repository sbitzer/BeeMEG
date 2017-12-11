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


#%% options

# label mode = mean
srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201712061743.h5'

srcfile = os.path.join(ss.bem_dir, srcfile)

area = 'v23ab'

normsrc = True

labels = ['L_%s_ROI-lh' % area, 'R_%s_ROI-rh' % area]

delay = 400

timeslice = [-1000, delay]

resptimes = pd.Index(np.arange(timeslice[0], timeslice[1]+10, step=10), 
                               name='time')

#%% example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')
times = epochs.index.levels[2]


#%% define the function which reindexes time to response-aligned
epochtimes = epochs.index.levels[2]
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
    
    # get response-aligned times of data points
    epochs.index = rtindex(sub)
    
    # exclude trials which were timed out (were coded as RT=5000 ms)
    epochs = epochs.loc[(rts.loc[sub][rts.loc[sub] <= 2500].index.values, 
                         slice(None))]
    
    return epochs

print("loading ... ", end='')
alldata = pd.concat([load_subject(sub) for sub in subjects],
                    keys=subjects, names=['subject', 'trial', 'time'])
print('done.')

if normsrc:
    epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
    epochs_std = pd.read_hdf(srcfile, 'epochs_std')
    
    alldata = (alldata - epochs_mean[labels]) / epochs_std[labels]

    
#%% create response-aligned sum_dot_x regressor

# load first dot onset aligned, time indexed sum_dot_x values
sumx = regressors.sumx_time(times.values / 1000, delay / 1000)

# exclude trials in which there was no response (timed out trials)
def get_response_trials(sub):
    sumxsub = sumx.loc[sub]
    rtssub = rts.loc[sub]
    return sumxsub.loc[(rtssub[rtssub != 5000].index, slice(None))]

sumx = pd.concat([get_response_trials(sub) for sub in subjects],
                 keys=subjects, names=['subject', 'trial', 'time'])

# now data and regressor are in the same order, so simply assign the 
# response-aligned index of alldata to the regressor values to get 
# response-aligned regressor values
sumx.index = alldata.index

# count the trials in which we have a regressor value at the chosen times
trial_counts = pd.concat([
        sumx.loc[sub].groupby(level='time').count() 
        for sub in subjects], keys=subjects, names=['subject', 'time'])

sumx_normalised = subject_DM.normfun(sumx)
    

#%% check relationship at particular time point before response
time = -650

data = pd.concat([sumx.xs(time, level='time'), alldata.xs(time, level='time')],
                 axis=1)
data = data.dropna()

color = 'k'
xrange = data.sum_dot_x.quantile([0.1, 0.9])
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 4.5))
for i in range(2):
    gpm = ss.gpregplot('sum_dot_x', labels[i], data, x_estimator=np.mean, x_bins=8,
                       line_kws={'color': color, 'label': '%d ms' % time},
                       scatter_kws={'color': color}, xrange=xrange, ax=axes[i])
    

#%% standard summary statistics inference
first_level = pd.DataFrame([],
        index=pd.MultiIndex.from_product(
                [labels, resptimes], names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [subjects, ['beta', 'bse'], ['intercept', 'sum_dot_x']], 
                names=['subject', 'measure', 'regressor']), dtype=float)
first_level.sort_index(axis=1, inplace=True)

second_level = pd.DataFrame([], 
        index=pd.MultiIndex.from_product(
                [labels, resptimes], names=['label', 'time']), 
        columns=pd.MultiIndex.from_product(
                [['mean', 'std', 'tval', 'mlog10p'], 
                 ['intercept', 'sum_dot_x']], 
                names=['measure', 'regressor']), dtype=np.float64)
second_level.sort_index(axis=1, inplace=True)

for label in labels:
    for time in resptimes:
        print('\rlabel %s, time = % 5d' % (label, time), end='', flush=True)
        
        data = pd.concat([sumx_normalised.xs(time, level='time'), 
                          alldata.xs(time, level='time')],
                 axis=1)
        data = data.dropna()
        
        for sub in subjects:
            datasub = data.loc[sub]
            if datasub.size > 10:
                res = sm.OLS(
                        datasub[label].values, 
                        np.c_[datasub.sum_dot_x.values, np.ones(datasub.shape[0])], 
                        hasconst=True).fit()
                first_level.loc[(label, time), (sub, slice(None), 'sum_dot_x')] = [
                        res.params[0], res.bse[0]]
                first_level.loc[(label, time), (sub, slice(None), 'intercept')] = [
                        res.params[1], res.bse[1]]
        
        params = first_level.loc[(label, time), 
                                 (slice(None), 'beta', slice(None))]
        params = params.unstack('regressor')
        
        params.dropna()
        
        second_level.loc[(label, time), ('mean', slice(None))] = (
                params.mean().values)
        second_level.loc[(label, time), ('std', slice(None))] = (
                params.std().values)
        tvals, pvals = ttest_1samp(params.values, 0, axis=0)
        second_level.loc[(label, time), ('tval', slice(None))] = tvals
        second_level.loc[(label, time), ('mlog10p', slice(None))] = (
                -np.log10(pvals))

print('')


#%% plot results
show_measure = 'mean'
alpha = 0.01

sl = second_level.dropna().copy()
ss.add_measure(sl, 'mlog10p_fdr')

r_colors = {'intercept': 'C0', 'sum_dot_x': 'C1'}
r_labels = {'intercept': 'average', 'sum_dot_x': 'sum x'}

fig, axes = plt.subplots(1, len(labels), sharey=True, figsize=(7.5, 4))

for label, ax in zip(labels, axes):
    lines = {}
    for r_name in sl.columns.levels[1]:
        lines[r_name], = ax.plot(sl.loc[label].index, 
                                 sl.loc[label].loc[:, (show_measure, r_name)], 
                                 label=r_labels[r_name], 
                                 color=r_colors[r_name])
    
    ti = sl.loc[label].index
    
    # determine whether positive or negative peaks are larger
    mm = pd.Series([sl.loc[label].loc[:, (show_measure, 'sum_dot_x')].min(),
                    sl.loc[label].loc[:, (show_measure, 'sum_dot_x')].max()])
    mm = mm[mm.abs().idxmax()]
    
    # plot markers over those time points which are significant after FDR
    significant = np.full(ti.size, mm * 1.1)
    significant[sl.loc[label, ('mlog10p_fdr', 'sum_dot_x')] 
                < -np.log10(alpha)] = np.nan
    ax.plot(ti, significant, '*', label='significant',
            color=lines['sum_dot_x'].get_color())
    ax.plot(ti[[0, -1]], [0, 0], '--k')
    
    ax.set_title(label[0])
    ax.set_xlabel('time from response (ms)')

yl = ax.get_ylim()
for ax in axes:
    ax.plot([0, 0], yl, ':k', label='response', zorder=1)
ax.set_ylim(yl)

ax.legend()

fig.savefig(os.path.join(helpers.figdir, 
                         'response-aligned_sum_dot_x_%s.png' % area), 
            dpi=300)


#%% plot number of data points per subject for plotted times
tmin = sl.index.get_level_values('time').min()
tmax = sl.index.get_level_values('time').max()

fig, ax = plt.subplots(1, figsize=(7.5, 4))

cnts = trial_counts.loc[(slice(None), slice(tmin, tmax))].unstack('subject')

lines = ax.plot(cnts.index, cnts.values, color=r_colors['sum_dot_x'], 
                alpha=0.5)
ax.set_ylabel('trial count')
ax.set_xlabel('time from response (ms)')
ax.set_title(r_labels[r_name])
ax.legend([lines[0]], ['single subjects'], loc='center right')

fig.savefig(os.path.join(helpers.figdir, 
                         'response-aligned_sum_dot_x_%s_trcounts.png' % area), 
            dpi=300)

#%% estimate second-level time course with GPs
color = 'k'
xrange = [-800, -100]
bin_width = 50
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 4))

fl = (first_level.dropna().xs('beta', level='measure', axis=1)
      .stack('subject').reset_index('time'))
fl = fl[(fl.time >= xrange[0]) & (fl.time <= xrange[1])]

r_names = ['sum_dot_x']

for label, ax in zip(labels, axes):
    for r_name in r_names:
        color = r_colors[r_name]
        r_label = r_labels[r_name]
        
        gpm = ss.gpregplot(
                'time', r_name, fl.loc[label], x_estimator=np.mean, 
                x_bins=int((xrange[1] - xrange[0]) / bin_width), 
                line_kws={'color': color, 'label': r_label},
                scatter_kws={'color': color}, xrange=xrange, ax=ax,
                gp_kws={'gptype': 'heteroscedastic', 'smooth': False})
    
    ax.plot(xrange, [0, 0], '--k')
    ax.set_ylabel('')
    ax.set_title(label[0])
    ax.set_xlabel('time from response (ms)')
    
if xrange[1] >= 0:
    yl = axes[0].get_ylim()
    for ax in axes:
        ax.plot([0, 0], yl, ':k', label='response', zorder=1)
    axes[0].set_ylim(yl);

axes[0].set_ylabel('estimated second-level beta')
axes[1].legend()

fig.savefig(os.path.join(helpers.figdir, 
                         'response-aligned_sum_dot_x_%s_gp.png' % area), 
            dpi=300)


#%% 
grouped = fl.loc[label].groupby('time')
Y = grouped[r_name].mean()
X = Y.index.values
Zvar = grouped[r_name].var()
xpred = np.linspace(*xrange, num=200)[:, None]

fig, ax = plt.subplots()

ss.plot_gp_result(ax, xpred, gpm, r_colors['sum_dot_x'], 'sum x')

ax.errorbar(X, Y, np.sqrt(Zvar / grouped[r_name].count()), 
            color=r_colors['sum_dot_x'])


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