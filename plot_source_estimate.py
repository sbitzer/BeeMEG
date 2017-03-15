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

from mayavi import mlab

subjects_dir = 'mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')

if len(sys.argv) < 2:
    r_name = 'dot_x'
else:
    r_name = sys.argv[1]

# find files
files = glob.glob(os.path.join(bem_dir, '*_source_' + r_name + '-lh.stc'))

if len(files) > 0:
    print "showing file: " + files[0]
    stc = mne.read_source_estimate(files[0])

    brain = stc.plot(subject='fsaverage', surface='white', time_viewer=True, hemi='both')
    brain.show_view('lateral')
    
    mlab.show()
else:
    raise ValueError('No appropriate source file found!')