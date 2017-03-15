#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:02:56 2017

@author: bitzer
"""

import os
import mne
import helpers
import pandas as pd
import numpy as np


#%% options
inffile_base = 'meg_sequential_201703011624'
inffile = os.path.join(helpers.resultsdir, inffile_base+'.h5')
subjects_dir = 'mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')


#%% make forward model
# MNE uses the info only to get channel names, exclude bad channels, account 
# for 'comps', whatever these are, and include projections in the inverse 
# operator. Although the transformation of device to head coordinates of the 
# subject (dev_head_t) is stored in the info of the forward solution, it is not
# used during the computation of the forward solution - the trans-file is used
# for that. For the average subject I can, therefore, just read any info 
# containing the relevant channel information as long as it doesn't contain any
# projections - the infos of the motion corrected files fulfill that.
info = mne.io.read_info('/home/bitzer/proni/BeeMEG/MEG/Raw/bm02a/bm02a1_mc.fif')

fwdfile = os.path.join(bem_dir, 'fsaverage-oct-6-fwd.fif')
if os.path.isfile(fwdfile):
    fwd = mne.read_forward_solution(fwdfile)
else:
    transfile = os.path.join(bem_dir, 'fsaverage-trans.fif')
    if not os.path.isfile(transfile):
        # note that I had to change the path to fsaverage in this function, 
        # because it apparently used an old directory structure
        mne.create_default_subject(mne_root='/home/bitzer/mne-python', 
                                   fs_home='/media/bitzer/Data/freesurfer', 
                                   update=True, subjects_dir=subjects_dir)
        
    bemfile = os.path.join(bem_dir, 'fsaverage-inner_skull-bem-sol.fif')
    if os.path.isfile(bemfile):
        bem = mne.read_bem_solution(bemfile)
    else:
        model = mne.read_bem_surfaces(os.path.join(bem_dir, 
                'fsaverage-inner_skull-bem.fif'))
    
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(os.path.join(bem_dir, 
                'fsaverage-inner_skull-bem-sol.fif'), bem)
    
    srcfile = os.path.join(bem_dir, 'fsaverage-oct-6-src.fif')
    if os.path.isfile(srcfile):
        src = mne.read_source_spaces(srcfile)
    else:
        src = mne.setup_source_space(subject, spacing='oct6', 
                                     subjects_dir=subjects_dir, n_jobs=4)
    
    fwd = mne.make_forward_solution(info, transfile, src, bem, meg=True, 
            eeg=False, mindist=5.0, n_jobs=4)
    
    mne.write_forward_solution(fwdfile, fwd)
    

#%% load data
second_level = pd.read_hdf(inffile, 'second_level')

# load normalisation values to revert scaling 
means = pd.read_hdf(inffile, 'epochs_mean')
stds = pd.read_hdf(inffile, 'epochs_std')

# if the standard deviations are orders of magnitudes higher than in the femto
# range, assume that MNE used standard scaling during epochs.to_data_frame
if any(stds > 1e-10):
    scaling = 1e15
else:
    scaling = 1.0

# restrict forward model to magnetometers
fwd = mne.pick_types_forward(fwd, meg='mag', eeg=False)

# get covariance container
ad_hoc_cov = mne.make_ad_hoc_cov(info)

# get data container
evoked = helpers.load_evoked_container(
        window=second_level.index.levels[1][[0, -1]] / 1000)
# keep MNE from scaling covariances as I do everything manually below
evoked.nave = 1


#%% reconstruct source
for r_name in second_level.columns.levels[1]:
    # permutations did not affect intercept, hence I need to estimate noise
    # somehow differently - simply use a regressor with similarly large effects
    if r_name == 'intercept':
        r_name_noise = 'dot_x'
    else:
        r_name_noise = r_name
    data_noise = second_level.loc[(slice(1, None), slice(None), slice(None)), 
                                  ('mean', r_name_noise)]
    data_noise = data_noise.reset_index(level='channel').pivot(columns='channel')
    
    # make a noise cov by estimating from permutations
    noise_cov = ad_hoc_cov.copy()
    cov = np.diag(noise_cov.data)
    mag_chans = mne.pick_types(info, meg='mag')
    # ensures that code is only applied to magnetometers
    assert mag_chans[0] == mne.pick_channels(info['ch_names'], 
                                             [second_level.index.levels[1][0]])
    # rescale noise covariance into original physical units
    cov[np.ix_(mag_chans, mag_chans)] = (  data_noise.cov().values 
                                         * stds.values**2
                                         / scaling**2)
    noise_cov.update(data=cov, diag=False)
    
    # make inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
            info, fwd, noise_cov, loose=0.2, depth=0.8)
    
    # select data
    data = second_level.loc[(0, slice(None), slice(None)), ('mean', r_name)]
    data = data.reset_index(level='channel').pivot(columns='channel')
    
    # rescale data into original physical units
    data = data * stds.values / scaling
    
    # put data into evoked container
    evoked.data = data.values.T
    
    # lambda2 is related to the precision of the prior over source currents
    # MNE suggests to set it to 1 / SNR**2
    lambda2 = 1 / (data.std().mean() / (  data_noise.xs(1, level='permnr')
                                        * stds.values / scaling).std().mean() )**2

    # compute!
    stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator, lambda2, method='dSPM', pick_ori=None)
    
    # save
    file = os.path.join(bem_dir, inffile_base + '_source_' + r_name)
    stc.save(file)
