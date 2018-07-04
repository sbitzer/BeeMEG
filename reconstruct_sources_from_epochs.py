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
import source_statistics as ss
import gc


#%% options
# baseline period (set to None for no baseline correction)
bl = (-0.3, 0)
#bl = None

# desired sampling frequency of data
sfreq = 100

# smoothing frequency; the lower, the more smoothing; set to sfreq for no 
# smoothing, otherwise can be maximally sfreq/2
hfreq = sfreq

# time window within an epoch to consider
window = [0, 2.5]

# chtype=True for all MEG-channels, chtype='mag' for magnetometers
chtype = True

# source reconstruction method
method = 'dSPM'

# fixed orientation? see doc for mne.minimum_norm.make_inverse_operator
inv_fixed = 'auto'
inv_loose = 0.2
inv_depth = 0.8

# properties of forward solution for given reconstruction constraints
fwd_surf_ori = True
if inv_depth is None and inv_loose == 0:
    fwd_force_fixed = True
else:
    fwd_force_fixed = False

# maintain dipole orientations or only magnitudes? see doc for 
# mne.minimum_norm.apply_inverse_epochs, choose None for magnitude of vector
# choose 'normal' for magnitude perpendicular to cortical surface
pick_ori = 'normal'

# parcellation on which labels are based
parc = 'HCPMMP1_5_8'

# specific labels that should be stored, all others are discarded,
# use empty list to indicate that all labels should be stored
#sections = [6, 7, 8, 16]
sections = []
anames = ss.Glasser_areas[ss.Glasser_areas['main section'].apply(
        lambda r: r in sections)]['area name']
lselection = (  list(anames.apply(lambda n: 'L_%s_ROI-lh' % n))
              + list(anames.apply(lambda n: 'R_%s_ROI-rh' % n)))

# set to None to keep original source points, 
# otherwise see MNE's extract_label_time_course for available modes
label_mode = 'mean_flip'

options = pd.Series({'chtype': chtype, 'sfreq': sfreq, 'hfreq': hfreq, 
                     'method': method, 'fwd_surf_ori': fwd_surf_ori,
                     'fwd_force_fixed': fwd_force_fixed,
                     'inv_fixed': inv_fixed, 'inv_loose': inv_loose,
                     'inv_depth': inv_depth, 'pick_ori': pick_ori, 
                     'parc': parc, 'label_mode': label_mode})

# where to store
subjects_dir = 'mne_subjects'
file = pd.datetime.now().strftime('source_epochs_allsubs_'+parc+'_%Y%m%d%H%M'+'.h5')
file = os.path.join(subjects_dir, 'fsaverage', 'bem', file)

with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['options'] = options
    store['bl'] = pd.Series(bl)
    store['window'] = pd.Series(window)
    store['lselection'] = pd.Series(lselection)


#%% run source reconstruction separately for each subject
#   see the old reconstruct_sources.py for more information on individual steps
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both')
if lselection:
    labels = [l for l in labels if l.name in lselection]
labelnames = [l.name for l in labels]

trials = np.arange(480) + 1

fresh = True
for sub in subjects:
    print('\nprocessing subject %2d' % sub)
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
    fwd = mne.read_forward_solution(fwdfile)
    fwd = mne.convert_forward_solution(fwd, fwd_surf_ori, fwd_force_fixed)
    
    # restrict forward model to selected types
    fwd = mne.pick_types_forward(fwd, meg=chtype, eeg=False)
    
    # compute sensor covariance from signal in baseline period
    cov = mne.compute_covariance(epochs, tmin=-0.3, tmax=0)
    
    # make inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
            info, fwd, cov, loose=inv_loose, fixed=inv_fixed, 
            depth=inv_depth)
    
    # lambda2 is related to the precision of the prior over source currents
    # MNE suggests to set it to 1 / SNR**2 and uses SNR=3 as default in 
    # mne.minimum_norm.apply_inverse
    lambda2 = 1 / 3**2
    
    # select time points for which to do source estimation
    epochs = epochs.crop(*window)
    
    # initialise time course store, because we didn't know times before
    times = (epochs.times * 1000).round().astype(int)
    
    # compute! out comes a generator object through which you can loop, then
    # only when an individual element is accessed the minimum norm estimate 
    # will be generated - this way you never need to hold all results in memory
    # at once
    stcs_gen = mne.minimum_norm.apply_inverse_epochs(
            epochs, inverse_operator, lambda2, method=method, 
            pick_ori=pick_ori, return_generator=True)
    
    labels = mne.read_labels_from_annot(subject, parc=parc, hemi='both')
    if lselection:
        labels = [l for l in labels if l.name in lselection]
    
    # this number of vertices may be misleading, because it is independent of 
    # the defined source space which may have a much wider mesh, the actually
    # used number of source points per label would be len(vertno) below
    label_nv = pd.DataFrame(np.array([len(l) for l in labels])[None, :], 
                            index=pd.Index([sub], name='subject'), 
                            columns=labelnames, dtype=int)
    
    # select source points (vertices), if no aggregation within labels
    if label_mode is None:        
        label_tc = []
        for trial, stc in zip(trials, stcs_gen):
            tc_trial = []
            for label in labels:
                label_stc = stc.in_label(label)
                
                # column names in result data frame should contain vertex 
                # numbers
                if label.hemi == 'lh':
                    lnames = [label.name + '_%06d' % vno 
                              for vno in label_stc.lh_vertno]
                else:
                    lnames = [label.name + '_%06d' % vno 
                              for vno in label_stc.rh_vertno]
                
                tc_trial.append(pd.DataFrame(
                        label_stc.data.T, 
                        columns=pd.Index(lnames, name='label'), 
                        index=pd.MultiIndex.from_product(
                                [[sub], [trial], times], 
                                names=['subject', 'trial', 'time']), 
                        dtype=float))
            
            label_tc.append(pd.concat(tc_trial, axis=1))
        
        # just using label_tc as name to maintain backward compatibility
        label_tc = pd.concat(label_tc)
    else:
        label_tc = pd.DataFrame([], 
                                columns=pd.Index(labelnames, name='label'), 
                                index=pd.MultiIndex.from_product(
                                        [[sub], trials, times], 
                                        names=['subject', 'trial', 'time']), 
                                dtype=float)
        
        label_tc_gen = mne.extract_label_time_course(
                stcs_gen, labels, fwd['src'], mode=label_mode)
        for trial, tc in zip(trials, label_tc_gen):
            label_tc.loc[(sub, trial, slice(None)), :] = tc.T
        
    if fresh:
        mean = label_tc.mean()
    else:
        mean += label_tc.mean()
        
    with pd.HDFStore(file, mode='a', complib='blosc', complevel=7) as store:
        store.append('label_nv', label_nv)
        store.append('label_tc', label_tc)
        
    gc.collect()

        
#%% save mean and std across all data points for each label
mean = mean / len(subjects)

with pd.HDFStore(file, 'r+') as store:
    sub = subjects[0]
    std = ((store.select('label_tc', 'subject==sub') - mean)**2).sum()
    for sub in subjects[1:]:
        gc.collect()
        std += ((store.select('label_tc', 'subject==sub') - mean)**2).sum()
        
    std = np.sqrt(std / (store.get_storer('label_tc').nrows - 1.5))
    
    store['epochs_mean'] = mean
    store['epochs_std'] = std