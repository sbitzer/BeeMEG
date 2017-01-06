#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:40:06 2016

@author: bitzer
"""

import os
from scipy.io import loadmat

#%%
basedir = 'data'
behdatadir = os.path.join(basedir, 'meg_behav')

d = loadmat(os.path.join(behdatadir, '05-ind.mat'))

# li/ri - left/right response of subject
# vi - valid trials = RT>700ms and correct
# vli/vri - left/right valid trials of subject

# directory of pre-processed MEG data
megdatadir = os.path.join(basedir, 'meg_final_data')

m = loadmat(os.path.join(megdatadir, '20_sdat.mat'))

# smeg - sorted meg (480 trials x 306 channels x 301 time steps)
# 1 time step = 4 ms
# 301 time steps start at -0.3 s (before 1st dot onset) to 0.9 s
# ITI was 1.5-1.8 s ??
# 306 channels: grad1_1, grad2_1, mag_1, ..., grad1_i, grad2_i, mag_i, ..., 
#               grad1_102, grad2_102, mag_102
# grad units: T/m
# mag units: T
# 480 trials: in order of dot positions

# setc - (480 trials x 3 channels x 301 time steps)
# 3 channels: 1-2 EOG (electro-oculogram?), 3 ECG (heartbeat)

# MEG device = Elekta Vectorview 2005(?)

# "bad" channels: were marked online and corrected (interpolation?)

"""
raw data files:
proni/BeeMEG/MEG/Raw
bm[sub]a[i].fif - raw data; sub is subject number and i is block number within experiment
bm[sub]a[i]_mc.fif - raw data after Maxfilter (signal space separation algorithm)

"""

"""
Matlab files for data processing:
    filt_dsample_meg.m ([sub]_down.mat)
 -> ICA_meg.m ([sub]_downi.mat)
 -> get_epochs.m ([sub]_ficai.mat)
 -> ICA_meg_sorttrials.m ([sub]_sdat.mat)
 
for converting to SPM format:
    scripts/convert_data.m
    
for getting model predictions into SPM:
    make_covariates.m
    
for preparing everything to pass into SPM:
    send2spm.m

example SPM scripts in folder scripts:
    preproc.m
"""

"""
original presentation log-files in _experiment/MEG_logfiles
processing of behavioural data:
    get_behav_main_megp.m
"""