#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:20:27 2017

@author: bitzer
"""

import os
import mne
import helpers
import pandas as pd
import numpy as np


#%% options
# baseline period (set to None for no baseline correction)
bl = (-0.3, 0)

# desired sampling frequency of data
sfreq = 100

# smoothing frequency; the lower, the more smoothing; set to sfreq for no 
# smoothing, otherwise can be maximally sfreq/2
hfreq = sfreq

# time window within an epoch to consider
window = [0, 0.9]

# chtype=True for all MEG-channels, chtype='mag' for magnetometers
chtype = True

# source reconstruction method
method = 'dSPM'

fwd_surf_ori = True

# fixed orientation?
fixed_ori = True

if fixed_ori:
    inv_fixed = True
    inv_loose = None
else:
    inv_fixed = False
    inv_loose = 0.2
    
inv_depth = 0.8

parc = 'HCPMMP1'

label_mode = 'max'

options = pd.Series({'chtype': chtype, 'sfreq': sfreq, 'hfreq': hfreq, 
                     'method': method, 'fwd_surf_ori': fwd_surf_ori,
                     'inv_fixed': inv_fixed, 'inv_loose': inv_loose,
                     'inv_depth': inv_depth, 'parc': parc,
                     'label_mode': label_mode})

# where to store
subjects_dir = 'mne_subjects'
file = pd.datetime.now().strftime('source_epochs_allsubs_'+parc+'_%Y%m%d%H%M'+'.h5')
file = os.path.join(subjects_dir, 'fsaverage', 'bem', file)

with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['options'] = options
    store['bl'] = pd.Series(bl)
    store['window'] = pd.Series(window)


#%% run source reconstruction separately for each subject
#   see the old reconstruct_sources.py for more information on individual steps
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both')
labelnames = [l.name for l in labels]

trials = np.arange(480) + 1

fresh = True
for sub in subjects:
    print('processing subject %2d' % sub)
    print('----------------------')
    
    # bem directory of this subject
    subject = 'bm%02d' % sub
    bem_dir = os.path.join(subjects_dir, subject, 'bem')
    
    info = mne.io.read_info('/home/bitzer/proni/BeeMEG/MEG/Raw/{0}a/'
                            '{0}a1_mc.fif'.format(subject))
    
    # load MNE epochs
    epochs = helpers.load_meg_epochs_from_sdat(sub, helpers.megdatadir)
    
    # pick specific channels
    epochs = epochs.pick_types(meg=chtype)
    
    # correct for baseline
    if bl is not None:
        epochs = epochs.apply_baseline(bl)
    
    # resample
    if sfreq != epochs.info['sfreq']:
        epochs = epochs.resample(sfreq)
    
    # smooth
    if hfreq < sfreq:
        epochs.savgol_filter(hfreq)
    
    # load forward operator
    fwdfile = os.path.join(bem_dir, '%s-oct-6-fwd.fif' % subject)
    fwd = mne.read_forward_solution(fwdfile, surf_ori=fwd_surf_ori)
    
    # restrict forward model to selected types
    fwd = mne.pick_types_forward(fwd, meg=chtype, eeg=False)
    
    # compute sensor covariance from signal in baseline period
    cov = mne.compute_covariance(epochs, tmin=bl[0], tmax=bl[1])
    
    # make inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
            info, fwd, cov, loose=inv_loose, fixed=inv_fixed, 
            depth=inv_depth)
    
    # lambda2 is related to the precision of the prior over source currents
    # MNE suggests to set it to 1 / SNR**2
    lambda2 = 1 / 3**2
    
    # select time points for which to do source estimation
    epochs = epochs.crop(*window)
    
    # initialise time course store, because we didn't know times before
    times = (epochs.times * 1000).round().astype(int)
    label_tc = pd.DataFrame([], 
                            columns=pd.Index(labelnames, name='label'), 
                            index=pd.MultiIndex.from_product(
                                    [[sub], trials, times], 
                                    names=['subject', 'trial', 'time']), 
                            dtype=float)

    # compute!
    stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inverse_operator, lambda2, method=method, pick_ori=None)
    
    labels = mne.read_labels_from_annot(subject, parc=parc, hemi='both')
    
    label_nv = pd.DataFrame(np.array([len(l) for l in labels])[None, :], 
                            index=pd.Index([sub], name='subject'), 
                            columns=labelnames, dtype=int)
    
    for trial, stc in zip(trials, stcs):
        label_tc.loc[(sub, trial, slice(None)), :] = (
                stc.extract_label_time_course(
                        labels, fwd['src'], mode=label_mode).T)
        
    if fresh:
        mean = label_tc.mean()
    else:
        mean += label_tc.mean()
        
    with pd.HDFStore(file, mode='a', complib='blosc', complevel=7) as store:
        store.append('label_nv', label_nv)
        store.append('label_tc', label_tc)

        
#%% save mean and std across all data points for each label
mean = mean / len(subjects)

with pd.HDFStore(file, 'r+') as store:
    sub = subjects[0]
    std = ((store.select('label_tc', 'subject==sub') - mean)**2).sum()
    for sub in subjects[1:]:
        std += ((store.select('label_tc', 'subject==sub') - mean)**2).sum()
        
    std = np.sqrt(std / (store.get_storer('label_tc').nrows - 1.5))
    
    store['epochs_mean'] = mean
    store['epochs_std'] = std