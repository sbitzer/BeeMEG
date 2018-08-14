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
import mne
import source_statistics as ss
import scipy.stats
import statsmodels.api as sm

# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, 
# choice flipped dot_x and accev, move_dist, sum_dot_y, constregs=0 for 1st dot
#resfile = 'meg_sequential_201707031205.h5'

# baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
resfile = 'meg_sequential_201802061518.h5'

# baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, sum_dot_x without dot_x, sum_dot_y_prev, percupt, 
# constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
#resfile = 'meg_sequential_201808091710.h5'

# should be chosen from the above, used for figures comparing momentary and
# accumulated evidence using separate regressions using dot_x or sum_dot_x
files = {'dot_x': 'meg_sequential_201802061518.h5',
         'sum_dot_x': 'meg_sequential_201808091710.h5'}

resfile = os.path.join(helpers.resultsdir, resfile)

figdir = helpers.figdir

accev_off = 100
accev_sign = -1


#%% helper functions
def plot_single_signal(fl_data, sl_data, label, r_name, ax=None, measure='mean'):
    sign = accev_sign if r_name == 'accev' else 1
    off = accev_off if r_name == 'accev' else 0
    
    dat = sign * fl_data.loc[(0, label, slice(None)), 
                             (slice(None), 'beta', r_name)].copy()
    times = dat.index.get_level_values('time')
    
    if ax is None:
        fig, ax = plt.subplots()
        
    l = ax.plot(times + off, dat, color='.7', lw=1, label='single subjects')
    l1 = ax.plot(
            times + off, sign * sl_data.loc[
                    (slice(1,3), label, slice(None)), (measure, r_name)]
                    .reset_index('permnr').pivot(columns='permnr'), 
            ':k', label='mean (permuted data)')
    l2 = ax.plot(times + off, sign * sl_data.loc[
            (0, label, slice(None)), (measure, r_name)], 
            'k', lw=2, label='mean')
    
    ax.legend([l[0], l1[0], l2[0]], ['single subjects', 'mean (permuted)', 'mean']);
    
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_ylabel(r'$\beta$')


#%% load sensor-level results
first_level = pd.read_hdf(resfile, 'first_level')
second_level = pd.read_hdf(resfile, 'second_level')
perms = second_level.index.levels[0]

evoked = helpers.load_evoked_container(window=[0, 0.9])

vmin = -0.1
vmax = -vmin


#%% average signals for dot_x and dot_y across time
fig = plt.figure(figsize=(7.5, 4.5))
xav_ax = plt.subplot(2, 2, 1)
yav_ax = plt.subplot(2, 2, 2, sharey=xav_ax, sharex=xav_ax)

x_times = np.array([120, 180, 250, 330, 400, 490, 570])
y_times = np.array([140, 200, 330, 390, 640])
T = len(x_times) + len(y_times)

xt_axes = [plt.subplot(2, T, T+p+1) for p in range(len(x_times))]
yt_axes = [plt.subplot(2, T, T+len(x_times)+p+1) for p in range(len(y_times))]

nperm = 3

linecol = '#4c72b0'

# add absmean curves
values = second_level.xs(0)[('mean', 'dot_x')].abs().mean(level='time')
datal, = xav_ax.plot(values.index, values.values, color=linecol, lw=2)
xav_ax.set_title('evidence: x-coordinate', fontdict={'fontsize': 12})
xav_ax.set_xlabel('time from dot onset (ms)')
xav_ax.set_ylabel(r'mean magnitude of grand average $\beta$')

markers, = xav_ax.plot(x_times, values.loc[x_times], '.k', ms=10)

for perm in range(1, nperm+1):
    values = second_level.xs(perm)[('mean', 'dot_x')].abs().mean(level='time')
    perml, = xav_ax.plot(values.index, values.values, ':', color=linecol, lw=1)

values = second_level.xs(0)[('mean', 'dot_y')].abs().mean(level='time')
yav_ax.plot(values.index, values.values, color=linecol, lw=2)
yav_ax.set_title('control: y-coordinate', fontdict={'fontsize': 12})
yav_ax.set_xlabel('time from dot onset (ms)')

yav_ax.plot(y_times, values.loc[y_times], '.k', ms=10)

for perm in range(1, nperm+1):
    values = second_level.xs(perm)[('mean', 'dot_y')].abs().mean(level='time')
    yav_ax.plot(values.index, values.values, ':', color=linecol, lw=1)

yav_ax.legend([datal, perml, markers], ['data', 'permuted', 'topo times'])


# add topomaps
values = second_level.xs(0)[('mean', 'dot_x')]
ev = mne.EvokedArray(values.values.reshape(102, values.index.levels[1].size), 
                     evoked.info, tmin=values.index.levels[1][0], 
                     nave=480*5)

ev.plot_topomap(x_times/1000, scalings=1, vmin=vmin, vmax=vmax, 
                image_interp='nearest', sensors=False,
                units='beta', outlines='skirt', axes=xt_axes, colorbar=False);

values = second_level.xs(0)[('mean', 'dot_y')]
ev = mne.EvokedArray(values.values.reshape(102, values.index.levels[1].size), 
                     evoked.info, tmin=values.index.levels[1][0], 
                     nave=480*5)

ev.plot_topomap(y_times/1000, scalings=1, vmin=vmin, vmax=vmax, 
                image_interp='nearest', sensors=False,
                units='beta', outlines='skirt', axes=yt_axes, colorbar=False);
                

# manually tune axes positions
left_m = 0.1
right_m = 0.03
top_m = 0.06
mid_space = 0.1
bottom = 0.3
width = (1-left_m-right_m-mid_space) / 2

xav_ax.set_position([left_m, bottom, width, 1-bottom-top_m])
yav_ax.set_position([left_m+width+mid_space, bottom, width, 1-bottom-top_m])

t_height = 0.12
t_width = 0.1
t_bot = 0.02
t_space = -0.025
t_left_m = 0.005
xy_space = 0.05

for i, ax in enumerate(xt_axes):
    ax.set_position([t_left_m + i*(t_width+t_space), t_bot, t_width, t_height],
                    which='original')
    ax.set_title(ax.get_title(), fontdict={'fontsize': 10})
for i, ax in enumerate(yt_axes):
    ax.set_position([  t_left_m + len(x_times)*(t_width+t_space) + xy_space
                     + i * (t_width+t_space), t_bot, t_width, t_height],
                    which='original')
    ax.set_title(ax.get_title(), fontdict={'fontsize': 10})
    
    
fig.text(0.02, 0.95, 'A', weight='bold', fontsize=18)
fig.text(0.51, 0.95, 'B', weight='bold', fontsize=18)
    
# export
fig.savefig(os.path.join(figdir, 'sensorspace_absmean.png'),
            dpi=300)


#%% paired t-test for x vs y in late period
slabs = second_level.loc[
        (0, slice(None), slice(300, 500))]['mean'][['dot_x', 'dot_y']].abs()

ttval, tpval = scipy.stats.ttest_rel(slabs['dot_x'], slabs['dot_y'])
print('t-test: t = %5.2f, p=%e' % (ttval, tpval))

# Wilcoxon signed rank test, because differences are not Gaussian
wval, wpval = scipy.stats.wilcoxon(slabs['dot_x'] - slabs['dot_y'])
print('Wilcoxon-test: W = %d, p=%e' % (wval, wpval))


#%% average signal for response across time
fig = plt.figure(figsize=(7.5, 3.5))

nperm = 2

linecol = '#4c72b0'

# time offset for trial regressors when trialregs_dot=-1
t_off = 200

# which time to show as topoplot?
r_time = 780


# plot absmean curve
ax = fig.add_axes([0.11, 0.15, 0.65, 0.75])
values = second_level.xs(0)[('mean', 'response')].abs().mean(level='time')
datal, = ax.plot(values.index + t_off, values.values, color=linecol, lw=2)
ax.set_title('response')
ax.set_xlabel('time from first dot onset (ms)')
ax.set_ylabel('average absolute beta')

markers, = ax.plot(r_time, values.loc[r_time-t_off], '.k', ms=10)

for perm in range(1, nperm+1):
    values = second_level.xs(perm)[('mean', 'response')].abs().mean(level='time')
    perml, = ax.plot(values.index + t_off, values.values, ':', color=linecol, lw=1)

ax.legend([datal, perml, markers], ['data', 'permuted', 'topo time'],
          loc='upper left')


# topoplot
ax = fig.add_axes([0.79, 0.28, 0.18, 0.5])

values = second_level.xs(0)[('mean', 'response')]
values = values.reset_index(level='channel').pivot(columns='channel')
values.columns = values.columns.droplevel([0,1])
    
ev = mne.EvokedArray(values.values.T, evoked.info, 
                     tmin=t_off/1000, nave=480*5)

ev.plot_topomap(r_time/1000, scale=1, vmin=vmin, vmax=vmax, 
                image_interp='nearest', unit='beta', outlines='skirt', axes=ax, 
                colorbar=False);
                
fig.savefig(os.path.join(figdir, 'response_absmean.png'), 
            dpi=300)


#%% show example beta-trajectories for selected sensors

def sensor_signal_plot(r_name, sensor, time, title, axes=None):
    if axes is None:
        fig = plt.figure(figsize=(7.5, 3.5))
    
    sign = accev_sign if r_name == 'accev' else 1
    off = accev_off / 1000 if r_name == 'accev' else 0
    
    # plot topomap for overview of selected sensor
    values = sign * second_level.xs(0)[('mean', r_name)].copy()
    values = values.reset_index(level='label').pivot(columns='label')
    values.columns = values.columns.droplevel([0,1])
    
    chmask = np.zeros((values.columns.size, values.index.size), dtype=bool)
    chmask[values.columns.get_loc(sensor), :] = True
    
    ev = mne.EvokedArray(values.values.T, evoked.info, 
                         tmin=values.index[0] + off, nave=480*5)
    
    if axes is None:
        t_ax = plt.axes([0.02, 0.28, 0.18, 0.5])
    else:
        t_ax = axes[0]
        fig = t_ax.get_figure()
        
    ev.plot_topomap(time, scalings=1, vmin=vmin, vmax=vmax, 
                    image_interp='nearest', units='beta', outlines='skirt',
                    axes=t_ax, colorbar=False, mask=chmask, 
                    mask_params={'markersize': 10}, time_unit='ms');
                    
    # plot time courses
    if axes is None:
        ax = plt.axes([0.3, 0.15, 0.65, 0.75])
    else:
        ax = axes[1]
        
    plot_single_signal(first_level, second_level, sensor, r_name, ax);
    ax.set_title(title)
    ax.set_ylim(-0.3, 0.3)
    
    return fig, [t_ax, ax]


#%%
fig, axes = sensor_signal_plot('dot_y', 'MEG2321', 0.19, 'y-coordinate')
fig2, axes2 = sensor_signal_plot('dot_x', 'MEG2241', 0.17, 'x-coordinate')


#%% compare grand average accumulated vs momentary evidence
measure = 'mean'
nperm = 3
npre = 2

a_times = np.r_[20, 80, 120, 180, 320, 400, 490]

cols = dict(accev='C0', 
            dot_x=[str(c) for c in np.linspace(0.7, 0.9, npre + 1)])
labels = dict(accev='accumulated', dot_x='momentary')

fig = plt.figure(figsize=(7, 4.5))
ax = plt.subplot(2, 1, 1)

# load x-correlations, produce time shifted versions and plot them with fill
sl = pd.read_hdf(os.path.join(helpers.resultsdir, files['dot_x']),
                 'second_level')
xmags = sl.xs(0)[(measure, 'dot_x')].abs().mean(level='time')
xmags = pd.DataFrame(xmags.values, index=xmags.index, columns=[0])
for pre in range(1, npre+1):
    xmagsprev = pd.DataFrame(xmags.loc[slice(pre * 100, None), 0].values, 
                             index=np.arange(0, 700 - pre * 100, 10), 
                             columns=[-pre])
    xmags[-pre] = xmagsprev
del xmagsprev
xmags.fillna(0, inplace=True)
for i, x in enumerate(xmags.columns):
    xl = ax.fill_between(xmags.index, xmags[x], color=cols['dot_x'][i], 
                          alpha=0.9, zorder=x, label=labels['dot_x'])

# load sum_dot_x and plot
sl = pd.read_hdf(os.path.join(helpers.resultsdir, files['sum_dot_x']),
                 'second_level')
for perm in range(1, nperm+1):
    values = sl.xs(perm)[(measure, 'sum_dot_x')].abs().mean(level='time')
    perml, = ax.plot(values.index, values.values, ':', 
                     color=cols['accev'], lw=2, alpha=0.3, zorder=1)

values = sl.xs(0)[(measure, 'sum_dot_x')].abs().mean(level='time')
al, = ax.plot(values.index, values.values, 
              color=cols['accev'], lw=3, alpha=1, zorder=2)

markers, = ax.plot(a_times, values.loc[a_times], '.k', ms=10)

ax.set_title('evidence', fontdict={'fontsize': 12})
ax.set_xlabel('time from dot onset (ms)')
ax.set_ylabel(r'mean magnitude of grand average $\beta$')

ax.legend([xl, al, perml], [labels['dot_x'], labels['accev'], 'permuted'])

ax.set_xlim(0, 690)
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1])

# plot topographies
T = len(a_times)
at_axes = [plt.subplot(2, T, T+p+1) for p in range(T)]

values = sl.xs(0)[(measure, 'sum_dot_x')]
ev = mne.EvokedArray(
        values.values.reshape(102, values.index.levels[1].size), 
        evoked.info, nave=480*5, 
        tmin=values.index.levels[1][0])

ev.plot_topomap(a_times / 1000, scalings=1, vmin=vmin, vmax=vmax, 
                image_interp='nearest', sensors=False, time_unit='ms',
                units='beta', outlines='skirt', axes=at_axes, colorbar=False);

# tune axis positions
ax.set_position([0.12, 0.38, 0.85, 0.56])
for tax in at_axes:
    bbox = tax.get_position(True)
    tax.set_position([bbox.x0, -0.11, bbox.width, bbox.height], 
                     which='original')
    
fig.savefig(os.path.join(figdir, 'accev_sensor_absmean.png'))


#%% compare accumulated evidence vs. dot_x in one sensor
channel = 'MEG0731'
showtime = 0.12

xl = [100, 690]

fig, axes = plt.subplots(2, 2)

sensor_signal_plot('accev', channel, showtime, '', axes[0, :]);
sensor_signal_plot('dot_x', channel, showtime, '', axes[1, :]);

axes[0, 1].set_xlim(*xl)
axes[1, 1].set_xlim(*xl)

axes[0, 1].set_title(channel)

axes[0, 0].set_position([0.08, 0.6, 0.15, 0.3])
axes[1, 0].set_position([0.08, 0.16, 0.15, 0.3])
axes[0, 1].set_position([0.36, 0.55, 0.6, 0.38])
axes[1, 1].set_position([0.36, 0.11, 0.6, 0.38])

axes[0, 1].set_xlabel('')
axes[0, 1].set_xticklabels([])

axes[1, 1].legend_.remove()

fig.text(0.02, 0.19, 'x-coordinate', rotation='vertical', va='bottom', 
         ha='left', size='x-large')
fig.text(0.02, 0.64, 'accumulated', rotation='vertical', va='bottom', 
         ha='left', size='x-large')

fig.savefig(os.path.join(figdir, 'accev_x_%s.png' % channel))


#%% compare grand average of accev and sum_dot_x from different regressions
files = {'accev': 'meg_sequential_201802061518.h5',
         'sum_dot_x': 'meg_sequential_201808091710.h5'}

measure = 'mean'
nperm = 3
cols = {'accev': 'C4', 'sum_dot_x': 'C2'}
lws = {'accev': 2, 'sum_dot_x': 2}
labels = {'accev': 'up to previous', 'sum_dot_x': 'up to current'}

sx_times = np.r_[20, 80, 120, 180, 320]

fig = plt.figure(figsize=(5, 4.5))
ax = plt.subplot(2, 1, 1)

for reg in ['accev', 'sum_dot_x']:
    sl = pd.read_hdf(os.path.join(helpers.resultsdir, files[reg]), 
                     'second_level')
    
    off = 0 if reg == 'sum_dot_x' else accev_off
    
    for perm in range(1, nperm+1):
        values = sl.xs(perm)[(measure, reg)].abs().mean(level='time')
        perml, = ax.plot(values.index + off, values.values, ':', 
                         color=cols[reg], lw=lws[reg], alpha=0.3)

    values = sl.xs(0)[(measure, reg)].abs().mean(level='time')
    al, = ax.plot(values.index + off, values.values, color=cols[reg], 
                  lw=lws[reg], alpha=1, label=labels[reg])
    
    if reg == 'sum_dot_x':
        values = second_level.xs(0)[(measure, reg)]
        ev = mne.EvokedArray(
                values.values.reshape(102, values.index.levels[1].size), 
                evoked.info, nave=480*5, 
                tmin=values.index.levels[1][0] / 1000)
    
ax.set_xlim(0, 690)
ax.legend()

T = len(sx_times)
sx_axes = [plt.subplot(2, T, T+p+1) for p in range(T)]

ev.plot_topomap(sx_times / 1000, scalings=1, vmin=vmin, vmax=vmax, 
                image_interp='nearest', sensors=False, time_unit='ms',
                units='beta', outlines='skirt', axes=sx_axes, colorbar=False);
                
                
#%% try to explain grand average sum_dot_x as combination of current and 
# previous dot_x
measure = 'mean'
# set to None for checking the grand average magnitudes
#channel = 'MEG0731'
channel = None

files = {'dot_x': 'meg_sequential_201802061518.h5',
         'sum_dot_x': 'meg_sequential_201808091710.h5'}

npre = 2
xcols = plt.cm.Greens_r(np.linspace(0.3, 0.7, npre+1))
zorders = [3, 2, 1]

sl = pd.read_hdf(os.path.join(helpers.resultsdir, files['dot_x']), 
                 'second_level')
if channel is None:
    xmags = sl.xs(0)[(measure, 'dot_x')].abs().mean(level='time')
else:
    xmags = sl.xs(0).loc[channel, (measure, 'dot_x')]
xmags = pd.DataFrame(xmags.values, index=xmags.index, columns=[0])
for pre in range(1, npre+1):
    xmagsprev = pd.DataFrame(xmags.loc[slice(pre * 100, None), 0].values, 
                             index=np.arange(0, 700 - pre * 100, 10), 
                             columns=[-pre])
    xmags[-pre] = xmagsprev
del xmagsprev
xmags.fillna(0, inplace=True)

# load sum_dot_x as accumulated evidence
sl = pd.read_hdf(os.path.join(helpers.resultsdir, files['sum_dot_x']), 
                 'second_level')
if channel is None:
    amags = sl.xs(0)[(measure, 'sum_dot_x')].abs().mean(level='time')
else:
    amags = sl.xs(0).loc[channel, (measure, 'sum_dot_x')]

res = sm.OLS(amags.values, xmags.values).fit()

# show fit
fig, ax = plt.subplots()

ax.plot(amags.index, amags, lw=3, zorder=10)
for i, x in enumerate(xmags.columns):
    ax.fill_between(xmags.index, xmags[x], color=xcols[i], alpha=0.9, 
                    zorder=zorders[i])
#    ax.plot(xmags.index, res.params[i] * xmags[x], lw=1, color=xcols[i])

#ax.plot(xmags.index, np.dot(xmags, res.params), lw=2, color='C2')