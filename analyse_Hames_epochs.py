#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:05:37 2017

@author: bitzer
"""

import numpy as np
import mne
from scipy.io import loadmat
import helpers
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns


#%% prepare epochs
sub = 5

# example raw file of subject 2, block 1
rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm02a/bm02a1.fif'

# get MEG-info from raw file (for channel layout and names)
info = mne.io.read_info(rawfile)

# Hame has downsampled to 4ms per sample
info['sfreq'] = 1 / 0.004

# create events array
events = np.c_[np.arange(480), np.full(480, 301, dtype=int), 
               np.zeros(480, dtype=int)]

# type of event
event_id = {'dot_onset': 0}
               
# load Hame's epoched data
# conveniently, the order of the channels in the matlab data is the same as
# the order given by the raw fif-file (see channel_names.mat which I extracted
# from variable hdr in one of the files in meg_original_data)
mat = loadmat('data/meg_final_data/%02d_sdat.mat' % sub)

# construct full data array corresponding to the channels defined in info
data = np.zeros((480, 313, 301))
# MEG data
data[:, :306, :] = mat['smeg']
# EOG + ECG
data[:, 306:309, :] = mat['setc'][:, :3, :]
# triggers (STI101) - Hame must have added this manually as I can't see where
# it was added in her Matlab scripts, but the time-courses and values 
# definitely suggest that it's STI101
# NOT PRESENT FOR ALL SUBJECTS
if mat['setc'].shape[1] == 4:
    data[:, 310, :] = mat['setc'][:, 3, :]

# when does an epoch start? - 0.3 s before onset of the first dot
tmin = -0.3

# create MNE epochs
epochs = mne.EpochsArray(data, info, events, tmin, event_id)


#%% build parametric regressor quantifying how much the 5th dot supports the 
#  corrrect decision
dotpos = helpers.load_dots(dotdir='data')
trial_info = helpers.get_5th_dot_infos(dotpos)

names = ['intercept', 'support_correct']
design_matrix = np.c_[np.ones(480), trial_info['support_correct']]

design_matrix[:, 1] = design_matrix[np.random.randint(480, size=480), 1]
                      
lm = mne.stats.regression.linear_regression(epochs, design_matrix, names)

#%% plot topomap of result
support_correct = lm['support_correct']

ch_type = 'mag'
times = np.linspace(0, .7, 15)
times = 'peaks'

# plot p-values
fig = support_correct.mlog10_p_val.plot_topomap(ch_type=ch_type, size=1.5, 
    unit='-log10_p', vmin=0, vmax=4.0, scale=1, times=times);

# extract times from titles
times = [float(ax.get_title()[:-3]) / 1000 for ax in fig.axes[:-1]]

# plot t-values
support_correct.t_val.plot_topomap(ch_type=ch_type, size=1.5, unit='t', 
    vmin=-4.0, vmax=4.0, scale=1, times=times);

                                   
#%% check linear regression manually
evo = support_correct.mlog10_p_val
chname, timeind = evo.get_peak('mag', time_as_index=True)
chind = np.flatnonzero(np.array(evo.ch_names) == 'MEG1341')[0]

df = pd.DataFrame(np.c_[data[:, chind, timeind], design_matrix[:, 1]], 
                  columns=['data', 'support_correct'])

res = smf.ols('data ~ support_correct', data=df).fit()

print(res.summary())