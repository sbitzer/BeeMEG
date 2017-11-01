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


#%% options
# baseline period (set to None for no baseline correction)
bl = (-0.3, 0)

# desired sampling frequency of data
sfreq = 100

# smoothing frequency; the lower, the more smoothing; set to sfreq for no 
# smoothing, otherwise can be maximally sfreq/2
hfreq = sfreq

# time window within an epoch to consider
window = [0.3, 0.9]

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

# parcellation on which labels are based
parc = 'HCPMMP1'

# specific labels that should be stored, all others are discarded,
# use empty list to indicate that all labels should be stored
sections = [6, 8]
anames = ss.Glasser_areas[ss.Glasser_areas['main section'].apply(
        lambda r: r in sections)]['area name']
lselection = (  list(anames.apply(lambda n: 'L_%s_ROI-lh' % n))
              + list(anames.apply(lambda n: 'R_%s_ROI-rh' % n)))

# set to None to keep original source points, 
# otherwise see MNE's extract_label_time_course for available modes
label_mode = None

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
    
    # compute!
    stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inverse_operator, lambda2, method=method, pick_ori=None)
    
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
        point_tcs = []
        for label in labels:
            # this is only used to extract label vertices in source space
            stc = stcs[0]
            
            # select those vertices that are both in the label and the 
            # source space
            if label.hemi == 'lh':
                vertno = np.intersect1d(stc.lh_vertno, label.vertices)
                vertidx = np.searchsorted(stc.lh_vertno, vertno)
            elif label.hemi == 'rh':
                vertno = np.intersect1d(stc.rh_vertno, label.vertices)
                vertidx = (  stc.lh_vertno.size 
                           + np.searchsorted(stc.rh_vertno, vertno))
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            
            # create result data frame with vertex numbers in label names
            lnames = [label.name + '_%06d' % vno for vno in vertno]
            point_tc = pd.DataFrame([], 
                                    columns=pd.Index(lnames, name='label'), 
                                    index=pd.MultiIndex.from_product(
                                            [[sub], trials, times], 
                                            names=['subject', 'trial', 'time']), 
                                    dtype=float)
            
            # store source data in result data frame
            point_tc.loc[(sub, slice(None), slice(None)), :] = np.concatenate(
                    [stc.data[vertidx, :].T for stc in stcs])
            # alternatively:
            # point_tc.loc[(sub, slice(None), slice(None)), :] = np.concatenate(
            #         [stc.in_label(label).data.T for stc in stcs])
            
            point_tcs.append(point_tc)
        
        # just using label_tc as name to maintain backward compatibility
        label_tc = pd.concat(point_tcs, axis=1)
    else:
        label_tc = pd.DataFrame([], 
                                columns=pd.Index(labelnames, name='label'), 
                                index=pd.MultiIndex.from_product(
                                        [[sub], trials, times], 
                                        names=['subject', 'trial', 'time']), 
                                dtype=float)
                            
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