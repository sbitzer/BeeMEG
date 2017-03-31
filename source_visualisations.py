#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:46 2017

to interact with Mayavi's mlab call ipython using (requires wxpython):

    ETS_TOOLKIT=wx ipython --gui=wx

probably QT is better, but I couldn't get it to work properly
see http://docs.enthought.com/mayavi/mayavi/mlab.html#simple-scripting-with-mlab
for further information about the pitfalls in trying to run mlab interactively

@author: bitzer
"""

import os
import re
import mne
import numpy as np
import glob


subjects_dir = 'mne_subjects'
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')
fig_dir = os.path.join(subjects_dir, subject, 'figures')

resultid_re = re.compile(r'.*_(\d{12})_.*.stc')

view = {'lh': {'azimuth': -55.936889415680255,
               'elevation': 28.757539951598851,
               'distance': 441.80865427261426,
               'focalpoint': np.r_[1.97380947, -20.13409592, -19.44953706]}}


def load_source_estimate(r_name='dot_x', f_pattern=''):
    # find files
    files = glob.glob(os.path.join(  bem_dir, '*' + f_pattern 
                                   + r_name + '-lh.stc'))
    
    if len(files) > 0:
        print "using file: " + files[0]
        stc = mne.read_source_estimate(files[0])
        stc.resultid = resultid_re.match(files[0]).groups()[0]
        stc.r_name = r_name
    else:
        raise ValueError('No appropriate source file found!')
    
    return stc


def make_movie(stc, hemi='lh', td=10):
    brain = stc.plot(subject='fsaverage', surface='inflated', hemi=hemi)
    brain.show_view(view[hemi])
    
    brain.save_movie(os.path.join(fig_dir, 'source_'+stc.resultid+'_'
                                           +stc.r_name+'_'+hemi+'.mp4'), 
                                  time_dilation=td)
        
    return brain

    
def time_viewer(stc, hemi='both'):
    brain = stc.plot(subject='fsaverage', surface='inflated', 
                     time_viewer=True, hemi=hemi)
    
    return brain


if __name__ == '__main__':
    stc = load_source_estimate()
    brain = stc.plot(subject='fsaverage', surface='inflated', hemi='both')
    brain.set_time(0.17)