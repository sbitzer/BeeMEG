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

# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, 
# choice flipped dot_x and accev, move_dist, sum_dot_y, constregs=0 for 1st dot
#resfile = 'meg_sequential_201707031205.h5'

# baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
resfile = 'meg_sequential_201802061518.h5'

resfile = os.path.join(helpers.resultsdir, resfile)

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% helper functions
def plot_single_signal(fl_data, sl_data, label, r_name, ax=None, measure='mean'):
    dat = fl_data.loc[(0, label, slice(None)), 
                      (slice(None), 'beta', r_name)]
    times = dat.index.get_level_values('time')
    
    if ax is None:
        fig, ax = plt.subplots()
        
    l = ax.plot(times, dat, color='.7', lw=1, label='single subjects')
    l1 = ax.plot(times, sl_data.loc[(slice(1,3), label, slice(None)), (measure, r_name)]
                               .reset_index('permnr')
                               .pivot(columns='permnr'), 
                 ':k', label='mean (permuted data)')
    l2 = ax.plot(times, sl_data.loc[(0, label, slice(None)), (measure, r_name)], 
                 'k', lw=2, label='mean')
    
    ax.legend([l[0], l1[0], l2[0]], ['single subjects', 'mean (permuted)', 'mean']);
    
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_ylabel('beta values')


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

def sensor_signal_plot(r_name, sensor, time, title):
    fig = plt.figure(figsize=(7.5, 3.5))
    
    # plot topomap for overview of selected sensor
    values = second_level.xs(0)[('mean', r_name)]
    values = values.reset_index(level='channel').pivot(columns='channel')
    values.columns = values.columns.droplevel([0,1])
    
    chmask = np.zeros((values.columns.size, values.index.size), dtype=bool)
    chmask[values.columns.get_loc(sensor), :] = True
    
    ev = mne.EvokedArray(values.values.T, evoked.info, 
                         tmin=values.index[0], nave=480*5)
    
    t_ax = plt.axes([0.02, 0.28, 0.18, 0.5])
    ev.plot_topomap(time, scale=1, vmin=vmin, vmax=vmax, image_interp='nearest', 
                    unit='beta', outlines='skirt', axes=t_ax, colorbar=False,
                    mask=chmask, mask_params={'markersize': 10});
                    
    
    # plot time courses
    ax = plt.axes([0.3, 0.15, 0.65, 0.75])
    plot_single_signal(first_level, second_level, sensor, r_name, ax);
    ax.set_title(title)
    ax.set_ylim(-0.3, 0.3)
    
    # export
    fig.savefig(os.path.join(figdir, 'sensorsignal_%s_%s.png' % (r_name, sensor)), 
                dpi=300)
    
    return fig


fig = sensor_signal_plot('dot_y', 'MEG2321', 0.19, 'y-coordinate')
fig2 = sensor_signal_plot('dot_x', 'MEG2241', 0.17, 'x-coordinate')

