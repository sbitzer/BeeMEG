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
import pandas as pd


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


def make_stc(srcfile, measure, r_name=None, src_df=None, transform=None, 
             mask=None, mask_other=-1):
    file = os.path.join(bem_dir, srcfile)
    
    if srcfile.find('_slabs_') >= 0:
        slabsi = file.find('slabs_')
        r_n = file[slabsi+6:-3]
        if r_name is None:
            r_name = r_n
        elif r_name != r_n:
            raise ValueError("Provided source file name and name of desired "
                             "regressor are not compatible!")
        
        if src_df is None:
            src_df = pd.read_hdf(file, 'second_level_src')
            
        data = src_df[measure].reset_index(level='time')
        data = data.pivot(columns='time')
        if transform is not None:
            data = transform(data)

        if mask is not None:
            data.where(mask, other=mask_other, inplace=True)

        stc = mne.SourceEstimate(
            data.values, 
            vertices=[data.loc['lh'].index.values, 
                      data.loc['rh'].index.values],
            tmin=data.columns.levels[1][0] / 1000.,
            tstep=np.diff(data.columns.levels[1][:2])[0] / 1000.,
            subject='fsaverage')
        
    return stc


if __name__ == '__main__':
    import matplotlib.cm as cm
    
    masked_hot = np.r_[np.zeros((128, 4)), cm.hot(np.linspace(0, 1, 128))] * 255
    
    srcfile = 'source_allsubs_201703301614_slabs_accev.h5'
    src_df = pd.read_hdf(os.path.join(bem_dir, srcfile), 'second_level_src')
    stc_mupl = make_stc(srcfile, 'mu_p_large', src_df=src_df)
    stc_mut = make_stc(srcfile, 'mu_t', src_df=src_df)
    mask = stc_mupl.data > 0.95
#    mask = stc_mut.data > 3.5
    
    stc = make_stc(srcfile, 'consistency', src_df=src_df, mask=mask)
    
    brain = stc.plot(subject='fsaverage', surface='white', hemi='both',
                     colormap=masked_hot, transparent=False, 
                     clim={'kind': 'value', 'lims': [-1, 0, 1]},
                     initial_time=0.3, smoothing_steps=5)