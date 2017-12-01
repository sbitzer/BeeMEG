#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:01:44 2017

@author: bitzer
"""

from __future__ import print_function
import source_visualisations as sv
import source_statistics as ss
import os
import mne
import pandas as pd

from mayavi import mlab
from surfer import Brain

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% load data
r_names = ['dot_x', 'dot_y']
measure = 'tval'
make_figures = True

# vertices of pre-motor and motor areas, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201711271306.h5'

src_df = pd.concat(
        [ss.load_src_df(basefile, r_name, use_basefile=True) 
         for r_name in r_names],
        keys=r_names, names=['regressor', 'label', 'time'])

# this performs FDR-correction across all regressors, vertices and times
ss.add_measure(src_df, 'p_fdr')


#%% prepare plotting
def get_colorinfo(measure, src_df, fdr_alpha=0.01):
    # find measure value that is the first one with p-value equal or smaller 
    # than fdr_alpha
    pdiff = src_df.p_fdr - fdr_alpha
    try:
        fmid = src_df[pdiff <= 0].sort_values('p_fdr')[measure].abs().iloc[-1]
    except IndexError:
        print('No FDR-corrected significant effects!')
        if measure == 'tval':
            fmin = src_df[measure].abs().min()
            fmax = src_df[measure].abs().max()
            colorinfo = {'fmin': fmin, 
                         'fmid': fmax, 
                         'fmax': fmax + fmax - fmin,
                         'center': 0}
    else:
        if measure == 'tval':
            colorinfo = {'fmin': fmid / 2., 
                         'fmid': fmid, 
                         'fmax': src_df[measure].abs().max(),
                         'center': 0}
        
    return colorinfo

views = {'rh': ({'azimuth': -125, 'elevation': 94, 
                 'focalpoint': (10.8, -38.6, -12.7)}, 91, 167.),
         'lh': ({'azimuth': -52, 'elevation': 99, 
                 'focalpoint': (-3.8, -48.8, -14.4)}, -96, 167.)}

def add_V1V2_labels(brain, src_df):
    lnames = src_df.index.get_level_values('label').map(
            lambda l: l[:-7]).unique()
    
    for hemi in brain.geo.keys():
        labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_5_8', 
                                            hemi=hemi)
        
        hemilnames = lnames[list(lnames.map(
                lambda s: s[0] == hemi[0].upper()).values)]
        
        for label in labels:
            if label.name in hemilnames:
                brain.add_label(label, borders=True, hemi=hemi, alpha=0.6)


#%% plot the data
r_name = 'dot_x'
                
hemis = ['lh', 'rh']
brains = {}
for hemi in hemis:
    fig = mlab.figure(size=(500, 500))
    
    brains[hemi] = Brain(
            'fsaverage', hemi, 'inflated', cortex='low_contrast',
            subjects_dir=sv.subjects_dir, background='w', foreground='k',
            figure=fig)
    
    brains[hemi].show_view(*views[hemi])
    
    add_V1V2_labels(brains[hemi], src_df)

    sv.show_label_vertices(src_df.loc[r_name], brains[hemi], measure, 
                           **get_colorinfo(measure, src_df))


#%% save figures
if make_figures:
    #times = [400]
    times = src_df.index.get_level_values('time').unique()
    
    outfiles = []
    for time in times:
        files = {}
        for hemi, brain in brains.iteritems():
            brain.set_time(time)
        
            file = os.path.join(figdir, 'V1V2_vertices_%s_%s_%s_%d.png' 
                                % (r_name, measure, hemi, time))
            files[hemi] = file
            brain.save_image(file, antialiased=True)
            
        # stitch them together
        outfiles.append(os.path.join(figdir, 'V1V2_vertices_%s_%s_%d.png' % 
                                     (r_name, measure, time)))
        os.system("montage -tile 2x1 -geometry +0+0 %s %s" % (
            ' '.join([files['lh'], files['rh']]), outfiles[-1]))
    
    if len(times) > 1:
        import shutil
        for ind, file in enumerate(outfiles):
            shutil.copyfile(file, 'tmp_%d.png' % (ind+1))
        
        mvfile = os.path.join(figdir, 'V1V2_vertices_%s_%s.mp4' % 
                              (r_name, measure))
        os.system('avconv -f image2 -r 2 -i tmp_%%d.png -vcodec mpeg4 -y %s' 
                  % mvfile)
        
        os.system('rm tmp_*.png')