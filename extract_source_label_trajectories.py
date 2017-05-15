#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:51:19 2017

@author: bitzer
"""

import os
import re
import mne
import numpy as np
import pandas as pd
import helpers

subjects_dir = 'mne_subjects'
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

def find_available_r_names(subjstr, inffile_base, bemdir):
    _, _, files = next(os.walk(bemdir))
    
    r_names = []
    for file in files:
        match = re.match(r'{}_{}_source_([\w_]+)-lh.stc'.format(subjstr, 
                         inffile_base, bemdir), file)
        if match is not None:
            r_names.append(match.group(1))
            
    return r_names
    

def get(srcres_file, atlas_name='HCPMMP1', agg_mode='mean_flip', 
        subjects=subjects, r_names=None):
    subjects = np.atleast_1d(subjects)
    
    # get used options for desirec source result
    srcopt = pd.read_hdf(os.path.join(subjects_dir, 'fsaverage', 'bem', 
                                      srcres_file), 'options')
    inffile_base = srcopt.inffile_base
    
    # read labels for fsaverage to get number of labels for atlas
    labels = mne.read_labels_from_annot('fsaverage', parc=atlas_name, hemi='both')
    labelnames = [l.name for l in labels]
    
    # read a source estimate for first subject to get times
    subjstr = 'bm%02d' % subjects[0]
    bemdir = os.path.join(subjects_dir, subjstr, 'bem')
    
    if r_names is None:
        r_names = find_available_r_names(subjstr, inffile_base, bemdir)
    r_names = np.atleast_1d(r_names)
    
    stcfile = '{}_{}_source_{}'.format(subjstr, inffile_base, r_names[0])
    stc = mne.read_source_estimate(os.path.join(bemdir, stcfile))
    
    times = stc.times
    
    # label time courses for each subject
    label_tc = pd.DataFrame([], 
                            columns=pd.MultiIndex.from_product(
                                    [subjects, r_names], 
                                    names=['subject', 'regressor']), 
                            index=pd.MultiIndex.from_product(
                                    [labelnames, times], 
                                    names=['label', 'time']), 
                            dtype=float)
    
    # number of vertices in each label for each subject
    label_nv = pd.DataFrame([], columns=subjects, index=labelnames, dtype=int)
    
    for sub in subjects:
        subjstr = 'bm%02d' % sub
        
        labels = mne.read_labels_from_annot(subjstr, parc=atlas_name, 
                                            hemi='both')
        
        label_nv.loc[:, sub] = np.array([len(l) for l in labels])
        
        bemdir = os.path.join(subjects_dir, subjstr, 'bem')
        
        fwd = mne.read_forward_solution(os.path.join(
                bemdir, '{}-oct-6-fwd.fif'.format(subjstr)), 
                surf_ori=srcopt.fwd_surf_ori)
        
        for r_name in r_names:
            stcfile = '{}_{}_source_{}'.format(subjstr, inffile_base, r_name)
            stc = mne.read_source_estimate(os.path.join(bemdir, stcfile))
            
            label_tc.loc[:, (sub, r_name)] = stc.extract_label_time_course(
                    labels, fwd['src']).flatten()
            
    return label_tc, label_nv
            
            
if __name__ == '__main__':
    label_tc, label_nv = get('source_allsubs_201703301614.h5')
    
    with pd.HDFStore(
            os.path.join(subjects_dir, 'fsaverage', 'bem', 
                         'source_HCPMMP1_allsubs_201703301614.h5')) as store:
        store['first_level_src'] = label_tc
        store['label_nv'] = label_nv