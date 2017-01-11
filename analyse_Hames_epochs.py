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

#%% prepare epochs
sub = 4

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

lm = mne.stats.regression.linear_regression(epochs, design_matrix, names)

#%% plot topomap of result
def plot_topomap(x, unit):
    x.plot_topomap(ch_type='mag', scale=1, size=1.5, vmax=np.max,
                   unit=unit, times=np.linspace(0.4, 0.9, 10))

support_correct = lm['support_correct']
    
plot_topomap(support_correct.beta, unit='beta')
plot_topomap(support_correct.mlog10_p_val, unit='-log10 p')