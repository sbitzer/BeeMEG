#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:27:25 2018

@author: bitzer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers
import os
import seaborn as sns
import analyse_eog as aeog


#%% load data
# horizontal EOG events
fname = 'eog_201802081218.h5'

file = os.path.join(helpers.resultsdir, fname)

with pd.HDFStore(file, 'r') as store:
    params = store.scalar_params
    measures = store.measures
    trial_times = store.trial_times
    eog_events = store.eog_events
    

#%% show correlation between RTs and event times across all subjects
# ignore timed-out trials
goodind = trial_times.RT != helpers.toresponse[1]
grid = sns.jointplot(
        'RT', 'event_time', 
        data=trial_times[goodind],
        kind='hex')

ax = grid.ax_joint
ax.set_autoscale_on(False)
xl = ax.get_xlim()
yl = ax.get_ylim()
ax.plot([xl[0], min(xl[1], yl[1])], [xl[0], min(xl[1], yl[1])], '--', 
        color='0.3', label='identity');

grid.fig.savefig(os.path.join(helpers.figdir, 'h-eog-event-time_vs_RT.png'),
                 dpi=300)

print('stats for event time - RT:')
print((trial_times[goodind].event_time - trial_times[goodind].RT).describe())


#%% 
grid = sns.jointplot('r_times', 'r_signs', data=measures)

grid.ax_joint.set_xlabel('corr with RT')
grid.ax_joint.set_ylabel('corr with choice')

grid.fig.savefig(os.path.join(helpers.figdir, 'h-eog-event_correlations.svg'))

#%%
sub = 2
epochs = helpers.load_meg_epochs_from_sdat(sub)
eog_ev, eog, eog_amp = aeog.find_eog_events(epochs, 'EOG061', params.sacdur, 
                                            params.perc, params.maxnpeak, 
                                            params.exclude_edge)
fig, axes = aeog.show_events(eog, eog_ev, np.arange(10)+11, params.sacdur)


#%% show overall distribution of horizontal EOG event times
fig, ax = plt.subplots()

sns.distplot(eog_events.index.get_level_values('time') / 1000,
             ax=ax);
ax.set_ylabel('density')
ax.set_xlabel('time from first dot onset')
ax.set_title('horizontal EOG events (N = %d)' % eog_events.shape[0]);


#%% load data for evoked time course
basefile = 'evoked_source_201802091710.h5'
sl = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 'second_level')


#%% evoked time course
label = sl.loc[0].abs().groupby('label').mean().tval.idxmax()[0]

fig, ax = plt.subplots()

times = sl.index.levels[2]
l, = ax.plot(sl.loc[(0, label)].tval, lw=2, label='estimate')
ls = ax.plot(times, sl.loc[(slice(1, None), label), 'tval'].unstack('permnr'), 
        ':', lw=1, color='0.3')
ax.set_title(label)
ax.set_ylabel('evoked effect (t-val)')
ax.set_xlabel('time after response (ms)')

ls[0].set_label('permuted')
ax.legend()

fig.savefig(os.path.join(helpers.figdir, 'evoked_timecourse_tval.svg'))


#%% load data for EOG sequential analysis
# standard sequential analysis of both EOG signals
fname = 'meg_sequential_201802081546.h5'

second_level = pd.read_hdf(os.path.join(helpers.resultsdir, fname), 
                           'second_level')

#%% show results of sequential analysis for horizontal and vertical EOG
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

chs = second_level.index.levels[1]

for ch, row in zip(chs, axes):
    for r_name, ax in zip(['dot_x', 'dot_y'], row):
        data = second_level.loc[
                (slice(None), ch), ('tval', r_name)].unstack('permnr')
        times = data.index.levels[1]
        
        ls = ax.plot(times, data.loc[:, slice(1, None)], ':k', 
                     label='permuted', lw=1)
        l, = ax.plot(times, data[0].values, label='estimate', lw=2)

axes[1, 0].set_xlabel('time from dot onset (s)')
axes[1, 1].set_xlabel('time from dot onset (s)')
axes[0, 0].set_ylabel('horizontal EOG')
axes[1, 0].set_ylabel('vertical EOG')
axes[0, 0].set_title('x-coordinate')
axes[0, 1].set_title('y-coordinate')

axes[1, 0].legend([l, ls[0]], (l.get_label(), ls[0].get_label()))

fig.tight_layout()

fig.savefig(os.path.join(helpers.figdir, 'eog_sequential_dot_x_dot_y.svg'))