#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:38:57 2018

@author: bitzer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:49:57 2017

@author: bitzer
"""

import helpers
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import source_statistics as ss

import scipy.stats

figdir = helpers.figdir


#%% load source-level results
# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201707031206.h5'

# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201709061827.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201801261754.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
#basefile = 'source_sequential_201801291241.h5'

# loose source orientations, but only their overall magnitude
# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
#basefile = 'source_sequential_201807031144.h5'

# loose source orientations, but cortex normal currents
# label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
basefile = 'source_sequential_201807041930.h5'

regressors = ['dot_x', 'dot_y', 'accev']

measure = 'mean'

if measure in ['mu_mean', 'mu_p_large']:
    second_level = pd.concat([ss.load_src_df(basefile, reg) for reg in regressors],
                             axis=1, keys=regressors, names=['regressor', 'measure'])
    second_level = second_level.swaplevel(axis=1).sort_index(axis=1)
else:
    second_level_perms = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 
                               'second_level')
    second_level = second_level_perms.loc[0]
    
    P = second_level_perms.index.get_level_values('permnr').unique().size

first_level = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 
                          'first_level')


#%% helper plotting function
def plot_single_source_signal(r_name, label, ax, t_off=0, t_slice=None,
                              sign=1, color='k', measure='mean'):
    if t_slice is None:
        t_slice = slice(None)
    dat = sign * first_level.loc[(0, label, t_slice), 
                                 (slice(None), 'beta', r_name)]
    times = dat.index.get_level_values('time') + t_off
    
    l = ax.plot(times, dat, color=color, alpha=0.1, lw=1, 
                label='single subjects')
    
    if measure == 'mu_mean':
        # get mean beta of folded normal mixture
        dat_mu = second_level.loc[(label, t_slice), ('mu_mean', r_name)]
        dat_theta = second_level.loc[(label, t_slice), ('theta_mean', r_name)]
        
        mean_beta = dat_theta * dat_mu - (1 - dat_theta) * dat_mu
    else:
        mean_beta = sign * second_level.loc[(label, t_slice), (measure, r_name)]
    
    l1, = ax.plot(times, mean_beta, color=color, lw=2, label='mean')
    
    return l[0], l1


#%% average effects for dot_x and dot_y over time
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 3])

# dot_x
values = second_level.loc[:, (measure, 'dot_x')].abs().mean(level='time')
x_times = [120, 170, 330, 410]

ax = axes[0]
line, = ax.plot(values.index, values)
ax.plot(x_times, values.loc[x_times], '.k', ms=10)
ax.set_xlabel('time from dot onset (ms)')
if measure == 'mu_mean':
    ax.set_ylabel('average posterior mu')
elif measure == 'mean':
    ax.set_ylabel('average absolute beta')
    for p in range(1, 4):
        ax.plot(values.index, 
                second_level_perms.loc[p, (measure, 'dot_x')].abs().mean(level='time'),
                ':', color=line.get_color(), lw=1)
ax.set_title('x-coordinate', fontdict={'fontsize': 12})

# dot_y
values = second_level.loc[:, (measure, 'dot_y')].abs().mean(level='time')
y_times = [120, 190, 350]

ax = axes[1]
ax.plot(values.index, values)
ax.plot(y_times, values.loc[y_times], '.k', ms=10)
if measure == 'mean':
    for p in range(1, 3):
        pl, = ax.plot(values.index, 
                second_level_perms.loc[p, (measure, 'dot_y')].abs().mean(level='time'),
                ':', color=line.get_color(), lw=1)
    ax.legend([line, pl], ['data', 'permuted'])
ax.set_xlabel('time from dot onset (ms)')
ax.set_title('y-coordinate', fontdict={'fontsize': 12})

fig.subplots_adjust(left=0.1, bottom=0.18, right=0.97)

fig.savefig(os.path.join(figdir, 'av_mu_mean_source.png'), 
            dpi=300)


#%% average effects of all regressors
fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[8, 3])

regs = dict(abs=dict(x='abs_dot_x', y='abs_dot_y', name='absolute'),
            mom=dict(x='dot_x', y='dot_y', name='momentary'),
            perc=dict(x='percupt_x', y='percupt_y', name='perceptual update'),
            acc=dict(x='accev', y='sum_dot_y_prev', name='accumulated'))

cols = dict(x='C0', y='C1')

leglines = []
legreg = 'perc'
for ax, reg in zip(axes, ['mom', 'acc', 'abs', 'perc']):
    for coord in ['x', 'y']:
        values = second_level.loc[:, (measure, regs[reg][coord])].abs().mean(
                level='time')
        
        # shift accumulated evidence regressor results 100 ms to later to
        # correct for it being defined only up to the previous dot, i.e., 
        # accumulated evidence of the dot of interest should be 100 ms later
        # I'm assuming here that the shape of the results is the same for 
        # previous and current dot
        if reg == 'acc':
            plottimes = values.index + 100
        else:
            plottimes = values.index
        
        line, = ax.plot(plottimes, values, color=cols[coord], label=coord)
        if legreg == reg:
            leglines.append(line)
        
        for p in range(1, P):
            pl, = ax.plot(plottimes, 
                    second_level_perms.loc[p, (measure, regs[reg][coord])].abs(
                            ).mean(level='time'),
                    ':', color=line.get_color(), lw=1, label='permuted')
        
        if legreg == reg and coord == 'y':
            leglines.append(pl)
    
    if legreg == reg:
        ax.legend(leglines, [l.get_label() for l in leglines])
    ax.set_xlabel('time (ms)')
    ax.set_title(regs[reg]['name'])

axes[0].set_ylabel(r'mean magnitude of second-level $\beta$')

fig.tight_layout()

fig.savefig(os.path.join(helpers.figdir, 'av_source_timecourses.png'), dpi=300)


#%% print area names with largest correlation magnitudes for dot_x in timeslice
timeslice = slice(0, 500)

sorted_areas = second_level.loc[(slice(None), timeslice), 
                                ('tval', 'dot_x')].abs().mean(
                                level='label').sort_values(ascending=False)

for hemi in ['L', 'R']:
    print(hemi)
    print(', '.join(sorted_areas[sorted_areas.index.map(
            lambda s: s.startswith(hemi))].head(5).index.map(
                    lambda s: s[2:-7])))
    

#%% get tidy data of areas with largest effects for two regressors
timeslice = slice(0, 500)

r_names = ['dot_x', 'dot_y']
labels = pd.Series([second_level.loc[(slice(None), timeslice), (measure, r_name)].abs().mean(
        level='label').idxmax() for r_name in r_names], r_names)

sl = second_level.loc[(labels, timeslice), ([measure, 'mlog10p'], r_names)].stack(
        'regressor')

# get rid of the data that I don't want to show and shouldn't influence
# multiple comparison correction
for r_name, label in labels.iteritems():
    other = set(r_names).difference(set([r_name])).pop()
    
    sl.loc[(label, slice(None), other), measure] = np.nan
    
sl.dropna(inplace=True)

ss.add_measure(sl, 'p_fdr')

#sl['significant'] = sl.p_fdr < 0.01
sl['significant'] = sl.mlog10p > -np.log10(0.01)


#%% plot example time courses for the brain area which has the largest overall
#   average effect
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 3])

rlabels = dict(dot_x='evidence', dot_y='y-coordinate')

for r_name, ax in zip(r_names, axes):
    label = labels[r_name]
    
    l, l1 = plot_single_source_signal(r_name, label, ax, t_slice=timeslice);
    
    sig = sl.loc[(label, timeslice, r_name), 'significant']
    sig = sig.index.get_level_values('time')[sig]
    l2, = ax.plot(sig, 0.13 * np.ones_like(sig), '.', color='#8E3A59')
    
    xl = sl.loc[(label, timeslice, r_name)].index.get_level_values('time').unique()[[0, -1]]
    ax.plot(xl, np.r_[0, 0], ':k')
    
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_title(rlabels[r_name] + ' (%s-%s)' % (label[0], label[2:-7]))

ax.legend([l, l1, l2], ['participants', 'estimated mean', 'p < 0.01'], 
          loc='lower right');

axes[0].set_ylabel(r'$\beta$')
          
fig.tight_layout()

fig.text(0.09, 0.91, 'B', weight='bold', fontsize=18)
fig.text(0.545, 0.91, 'C', weight='bold', fontsize=18)

fig.savefig(os.path.join(figdir, 'example_area_tcs.svg'), 
            dpi=300)


#%% average effects for accev over time
fig, ax = plt.subplots(1, figsize=[4, 3])

values = second_level.loc[:, (measure, 'accev')].abs().mean(level='time')

x_times = [20, 150, 230, 300]

line, = ax.plot(values.index + 100, values)
ax.plot(np.array(x_times) + 100, values.loc[x_times], '.k', ms=10)
ax.set_xlabel('time from dot onset (ms)')
if measure == 'mu_mean':
    ax.set_ylabel('average posterior mu')
elif measure == 'mean':
    ax.set_ylabel('average absolute beta')
    for p in range(1, 3):
        ax.plot(values.index + 100, 
                second_level_perms.loc[p, (measure, 'dot_x')].abs().mean(level='time'),
                ':', color=line.get_color(), lw=1)
ax.set_title('accumulated evidence', fontdict={'fontsize': 12})

fig.subplots_adjust(top=0.89, bottom=0.16, left=0.19, right=0.96)
fig.savefig(os.path.join(figdir, 'av_mu_mean_source_accev.png'), 
            dpi=300)


#%% example time courses for accev
labels = ['L_v23ab_ROI-lh', 'R_v23ab_ROI-rh', 'L_4_ROI-lh']

fig, axes = plt.subplots(1, len(labels), sharex=True, sharey=True, 
                         figsize=[7.5, 3])

r_name = 'accev'

if r_name == 'accev':
    t_off = 100
else:
    t_off = 0

print(second_level.loc[:, (measure, r_name)].abs().mean(level='label')
      .sort_values(ascending=False).head(10))

for ax, label in zip(axes, labels):
    l, l1 = plot_single_source_signal(r_name, label, ax, t_off);
    xl = ax.get_xlim()
    ax.plot(xl, np.r_[0, 0], ':k')
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_ylabel('beta-values')
    ax.set_title('accumulated evidence (%s-%s)' % (label[0], label[2:-7]))

ax.set_xlim(second_level.index.levels[1][[0, -1]] + t_off)

ax.legend([l, l1], ['single subjects', 'estimated mean']);

fig.subplots_adjust(left=0.1, bottom=0.18, right=0.97)

fig.savefig(os.path.join(figdir, 'example_area_tcs_accev.png'), 
            dpi=300)


#%% compare example time courses for dot_x and accev
area = 'AIP'
showme = 'tval'

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.5, 3))

accev_off = 100
accev_sign = -1

cols = dict(dot_x='C0', accev='C1')

alpha = 0.01
tsig = scipy.stats.t.ppf(alpha/2, first_level.columns.levels[0].size - 1)

for ax, hemi in zip(axes, ['L', 'R']):
    label = '%s_%s_ROI-%sh' % (hemi, area, hemi.lower())
    
    data = second_level.loc[label, (showme, 'dot_x')]
    l1, = ax.plot(data.index, data, lw=2, color=cols['dot_x'])
    
    data = accev_sign * second_level.loc[label, (showme, 'accev')]
    l2, = ax.plot(data.index + accev_off, data, lw=2, color=cols['accev'])
    
    xl = ax.get_xlim()
    ax.fill_between(xl, tsig * np.ones(2), -tsig * np.ones(2), color='0.9',
                    alpha=0.8, zorder=10)
    ax.plot(xl, np.r_[0, 0], ':k')
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_title('%s-%s' % (hemi, area))

axes[0].set_ylabel('t-values')
axes[0].set_xlim(100, 600)

xl = axes[0].get_xlim()
yl = axes[0].get_ylim()
axes[0].text(xl[1] - 20, yl[0] + 0.3, 'p < %4.2f' % alpha, va='bottom', 
             ha='right')
axes[0].text(xl[1] - 20, tsig + 0.3, 'p > %4.2f' % alpha, va='bottom', 
             ha='right')

axes[1].legend((l1, l2), ['momentary', 'accumulated'])

fig.tight_layout()
    

#%% source effects for response over time
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(7.5, 3))

t_off = 200
r_name = 'response'

# plot absmean
values = second_level.loc[:, ('mu_mean', r_name)].mean(level='time')
r_times = np.array([780])

ax = axes[0]
ax.plot(values.index+t_off, values)
ax.plot(r_times, values.loc[r_times-t_off], '.k', ms=10)
ax.set_xlabel('time from first dot onset (ms)')
ax.set_ylabel('average posterior mu')
ax.set_title('response')

# plot max label
label = second_level.loc[:, ('mu_mean', r_name)].mean(level='label').idxmax()
#label = 'R_4_ROI-rh'
#label = 'L_4_ROI-lh'

ax = axes[1]
plot_single_source_signal(r_name, label, ax, t_off);
ax.plot(ax.get_xlim(), np.r_[0, 0], ':k')
ax.set_xlabel('time from first dot onset (ms)')
ax.set_ylabel('beta-values')
ax.set_title('response (%s-%s)' % (label[0], label[2:-7]))

fig.subplots_adjust(left=0.1, bottom=0.17, right=0.97, top=0.91, wspace=0.3)

fig.savefig(os.path.join(figdir, 'response_av_mu_mean_source_exmp_area.png'), 
            dpi=300)


#%% early effects in V1/V2 source dipoles / points

# GLM on source points of V1 and V2
# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, source GLM, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201708151458.h5'

regressors = ['dot_x', 'dot_y']

second_level = pd.concat([ss.load_src_df(basefile, reg) for reg in regressors],
                         axis=1, keys=regressors, names=['regressor', 'measure'])
second_level = second_level.swaplevel(axis=1).sort_index(axis=1)

# restrict to V1 or V2 only
#second_level = second_level.loc[
#        ([n for n in filter(lambda n: 'V2' in n, second_level.index.levels[0])], 
#         slice(None))]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=[7.5, 3])

# average posterior mu
values = pd.concat(
        [second_level.loc[:, ('mu_mean', 'dot_x')].mean(level='time'),
         second_level.loc[:, ('mu_mean', 'dot_y')].mean(level='time')], axis=1)

ax = axes[0]
ax.plot(values.index, values.values[:, 0], label='x-coordinate')
ax.plot(values.index, values.values[:, 1], label='y-coordinate')
ax.set_xlabel('time from dot onset (ms)')
ax.set_ylabel('posterior mu')
ax.set_title('average', fontdict={'fontsize': 12})
ax.legend(loc='upper left')

# dot_y
values = pd.concat(
        [second_level.loc[:, ('mu_mean', 'dot_x')].max(level='time'),
         second_level.loc[:, ('mu_mean', 'dot_y')].max(level='time')], axis=1)

ax = axes[1]
ax.plot(values.index, values)
ax.set_xlabel('time from dot onset (ms)')
ax.set_title('maximum', fontdict={'fontsize': 12})

fig.subplots_adjust(left=0.1, bottom=0.18, right=0.97)

fig.savefig(os.path.join(figdir, 'av_mu_mean_source_V1V2.png'), 
            dpi=300)