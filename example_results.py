#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:19:20 2017

@author: bitzer
"""

import matplotlib.pyplot as plt
import pandas as pd
import helpers
import scipy.signal
import numpy as np
import mne


resfile = helpers.resultsdir + '/meg_sequential_201703011624.h5'
erffile = helpers.resultsdir + '/evoked_meg_sequential_201703101837.h5'

old_log_level = mne.set_log_level('WARNING', True)
evoked = helpers.load_evoked_container(window=pd.read_hdf(resfile, 'window'))
mne.set_log_level(old_log_level)
del old_log_level

second_level = pd.read_hdf(resfile, 'second_level')
perms = second_level.index.levels[0]

first_level = pd.read_hdf(resfile, 'first_level')

erf = pd.read_hdf(erffile, 'erf')
grand_erf = erf.mean(axis=1)

def plot_absmax_channel(r_name, time):
    channel = second_level.loc[(0, slice(None), time), ('mean', r_name)].abs().argmax()
    channel = channel[1]
    dat = first_level.loc[(0, channel, slice(None)), 
                          (slice(None), 'beta', r_name)]
    times = dat.index.get_level_values('time')
    fig, ax = plt.subplots()
    l = ax.plot(times, dat, color='.7', label='single subjects')
    l1 = ax.plot(times, second_level.loc[(slice(1,3), channel, slice(None)), ('mean', r_name)]
                                    .reset_index('permnr')
                                    .pivot(columns='permnr'), 
                 ':k', label='mean (permuted data)')
    l2 = ax.plot(times, second_level.loc[(0, channel, slice(None)), ('mean', r_name)], 
                 'k', lw=2, label='mean')
    l3 = ax.plot(times, grand_erf.loc[0, channel], '--k')
    
    ax.legend([l[0], l1[0], l2[0], l3[0]], ['single subjects', 'mean (permuted data)', 'mean', 'erf']);
    ax.set_title(channel)
    ax.set_xlabel('time from dot onset (ms)')
    ax.set_ylabel('beta of ' + r_name + ' (z)')
    

def show_topography_at_peaks(r_name, measure='mean', mode='absmax', order=4, vmin=-.1, vmax=.1):
    if mode == 'absmax':
        data = second_level.loc[0, (measure, r_name)].abs().max(level='time')
    elif mode == 'max':
        data = second_level.loc[0, (measure, r_name)].max(level='time')
    else:
        data = second_level.loc[0, (measure, r_name)].min(level='time').abs()
    
    fig, ax = plt.subplots()
    data.plot(ax=ax, title=measure, style='k', lw=2);
    
    # identify local maxima
    times = data.index[scipy.signal.argrelextrema(data.values, 
                                                  np.greater, order=order)] / 1000
    
    data = second_level.loc[0, (measure, r_name)]
    ev = mne.EvokedArray(data.values.reshape(102, data.index.levels[1].size), 
                         evoked.info, tmin=data.index.levels[1][0], 
                         nave=480*5, comment=r_name)
    
    fig = ev.plot_topomap(times, scale=1, vmin=vmin, vmax=vmax, image_interp='nearest', 
                          title=r_name+' aligned to dot onset', unit=measure, 
                          outlines='skirt');