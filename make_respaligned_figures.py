#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:51:13 2018

@author: bitzer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helpers
import source_statistics as ss
import os
import scipy.stats


#%% load data
resfile = 'source_response-aligned_201803221650.h5'

with pd.HDFStore(os.path.join(helpers.resultsdir, resfile), 'r') as store:
    first_level = store['first_level']
    second_level = store['second_level']
    data_counts = store['data_counts']

labels = second_level.index.get_level_values('label').unique()
area = labels[0][2:-7]

resp_align = resfile.find('response-aligned') > 0

r_names = second_level.columns.get_level_values('regressor').unique()


#%% average over delays and compute statistics

# get first level values as average over delays
fl = first_level.groupby(['label', 'time']).mean()

# re-compute second level stats
def statfun(row):
    betas = row.loc[(slice(None), 'beta', slice(None))].unstack(
            'regressor').dropna()
    stds = betas.std().values
    stes = stds / np.sqrt(betas.iloc[:, 0].count())
    
    tvals, pvals = scipy.stats.ttest_1samp(betas, 0, axis=0)
    
    return pd.Series(np.r_[betas.mean().values, stds, stes,
                           tvals, pvals, -np.log10(pvals)], 
                     index=pd.MultiIndex.from_product(
                             [['mean', 'std', 'ste', 'tval', 'pval', 'mlog10p'], 
                              row.index.levels[2]],
                             names=['measure', 'regressor']),
                     name=row.name)

sl = fl.apply(statfun, axis=1)

ss.add_measure(sl, 'mlog10p_fdr')


#%% define colors for plotting
if resp_align:
    fbase = 'response-aligned'
else:
    fbase = 'firstdot-aligned'
fbase += '_' + area

r_colors = {'intercept': 'C0', 'dot_x_time': 'C1', 'dot_y_time': 'C2', 
            'percupt_x_time': 'C3', 'percupt_y_time': 'C4'}
r_labels = {'percupt_y_time': 'PU-y', 
            'percupt_x_time': 'PU-x',
            'dot_x_time': 'evidence',
            'dot_y_time': 'y-coord',
            'intercept': 'intercept'}


#%% plot number of data points per subject for plotted times
fig, ax = plt.subplots(1, figsize=(7.5, 4))

cnts = data_counts.groupby('time').mean()

lines = ax.plot(cnts.index, cnts.values, color='k', alpha=0.5)

if resp_align:
    ax.set_xlabel('time from response (ms)')
else:
    ax.set_xlabel('time from first dot onset (ms)')
ax.set_ylabel('average data count across delays')
ax.legend([lines[0]], ['single subjects'], loc='center right')

fig.savefig(os.path.join(helpers.figdir, 'avdatacounts_%s.png' % fbase), dpi=300)


#%% plot results
show_measure = 'mean'
alpha = 0.01

# to get all significant effects:
print('all significant effects:')
print(sl.mlog10p_fdr.stack().groupby('regressor')
      .apply(lambda s: s[s > -np.log10(alpha)]))

r_name = 'dot_x_time'

# which regressors to show? 
# add main regressor of interest to end to show it on top
show_names = ['dot_y_time']
show_names.append(r_name)

leftright = dict(L='left', R='right')

fig, axes = plt.subplots(1, len(labels), sharey=True, figsize=(7.5, 4))

for label, ax in zip(labels, axes):
    lines = {}
    for rn in show_names:
        ax.fill_between(
                sl.loc[label].index, 
                sl.loc[label, ('mean', rn)] + 2 * sl.loc[label, ('ste', rn)], 
                sl.loc[label, ('mean', rn)] - 2 * sl.loc[label, ('ste', rn)], 
                alpha=0.2, color=r_colors[rn])
        lines[rn], = ax.plot(sl.loc[label].index, 
                                 sl.loc[label, ('mean', rn)], 
                                 label=r_labels[rn], color=r_colors[rn])
    
    ti = sl.loc[label].index
    
    # determine whether positive or negative peaks are larger
    mm = pd.Series([sl.loc[label].loc[:, ('mean', r_name)].min(),
                    sl.loc[label].loc[:, ('mean', r_name)].max()])
    mm = mm[mm.abs().idxmax()]
    
    # plot markers over those time points which are significant after FDR
    significant = np.full(ti.size, mm * 1.1)
    significant[sl.loc[label, ('mlog10p_fdr', r_name)] 
                < -np.log10(alpha)] = np.nan
    ax.plot(ti, significant, '.', label='significant',
            color=lines[r_name].get_color())
    ax.plot(ti[[0, -1]], [0, 0], '--k', zorder=1)
    
    title = 'primary motor cortex' if area == '4' else area
    ax.set_title(leftright[label[0]] + ' ' + title)
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

axes[0].set_ylabel(r'estimated second-level $\beta$')

fig.tight_layout()
fig.savefig(os.path.join(helpers.figdir, 
                         '%s_%s.png' % (r_name[:-5], fbase)), dpi=300)


#%% estimate second-level time course with GPs
#xrange = timeslice
#xrange = [-600, timeslice[-1]]
xrange = [-1000, -200]
bin_width = 100
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 5))

flgp = (fl.xs('beta', level='measure', axis=1)
      .stack('subject').reset_index('time'))
flgp = flgp[(flgp.time >= xrange[0]) & (flgp.time <= xrange[1])]

r_name = 'dot_x_time'
# which regressors to show? 
# add main regressor of interest to end to show it on top
show_names = ['dot_y_time']
show_names.append(r_name)

sigys = pd.Series(np.linspace(0.05 + (len(show_names) - 1) * 0.005, 0.05, 
                              len(show_names)), show_names)

for label, ax in zip(labels, axes):
    for rn in show_names:
        color = r_colors[rn]
        r_label = r_labels[rn]
        
        gpm, significant = ss.gpregplot(
                'time', rn, flgp.loc[label], x_estimator=np.mean, 
                x_bins=int((xrange[1] - xrange[0]) / bin_width), 
                plot_kws=dict(color=color, label=r_label, 
                              return_significant=True),
                scatter_kws={'color': color}, xrange=xrange, ax=ax,
                gp_kws={'gptype': 'heteroscedastic', 'smooth': False})
                
        sigy = sigys[rn]
        if label.startswith('L'):
            sigy *= -1
        ax.plot(significant, sigy * np.ones_like(significant), '.', 
                color=color)
    
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

