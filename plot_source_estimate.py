#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:05:08 2017

@author: bitzer
"""

import os
import mne
import pandas as pd
import numpy as np
import sys
import glob
import argparse
from mayavi import mlab

parser = argparse.ArgumentParser(description='Show reconstructed sources using pySurfer.')
parser.add_argument('r_name', nargs='?', default='dot_x',
                    help='the name of the regressor for which sources should be shown')
parser.add_argument('f_pattern', nargs='?', default='',
                    help='file pattern identifying the reconstructed sources (default: "")')

args = parser.parse_args()

subjects_dir = 'mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')

# find files
files = glob.glob(os.path.join(  bem_dir, '*' + args.f_pattern + '*' 
                               + '_source_' + args.r_name + '-lh.stc'))

if len(files) > 0:
    print "showing file: " + files[0]
    stc = mne.read_source_estimate(files[0])

    brain = stc.plot(subject='fsaverage', surface='inflated', time_viewer=True, hemi='both')
    brain.show_view('lateral')
    
    mlab.show(stop=True)
else:
    raise ValueError('No appropriate source file found!')