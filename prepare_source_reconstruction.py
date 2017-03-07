#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:02:56 2017

@author: bitzer
"""

import os
import mne

subjects_dir = 'mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')

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
    
    info = mne.io.read_info('/home/bitzer/proni/BeeMEG/MEG/Raw/bm02a/bm02a1.fif')
    
    srcfile = os.path.join(bem_dir, 'fsaverage-oct-6-src.fif')
    if os.path.isfile(srcfile):
        src = mne.read_source_spaces(srcfile)
    else:
        src = mne.setup_source_space(subject, spacing='oct6', 
                                     subjects_dir=subjects_dir, n_jobs=4)
    
    fwd = mne.make_forward_solution(info, transfile, src, bem, meg=True, 
            eeg=False, mindist=5.0, n_jobs=4)
    
    mne.write_forward_solution(fwdfile, fwd)