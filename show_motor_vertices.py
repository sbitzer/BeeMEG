#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:33 2017

@author: bitzer
"""

from __future__ import print_function
import source_visualisations as sv
import source_statistics as ss
import os
import mne

from surfer import Brain

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% load data
r_name = 'dot_x'
measure = 'tval'

# vertices of pre-motor and motor areas, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201710231824.h5'

src_df = ss.load_src_df(basefile, r_name, use_basefile=True)


#%% prepare plotting
def get_colorinfo(measure, src_df):
    if measure == 'tval':
        colorinfo = {'fmin': src_df[measure].abs().quantile(0.5), 
                     'fmid': src_df[measure].abs().quantile(0.95), 
                     'fmax': src_df[measure].abs().max(),
                     'center': 0}
        
    return colorinfo

views = {'rh': ({'azimuth': 6.8, 'elevation': 60.3, 
                 'focalpoint': (-0.60, 0., 0.)}, -90, 295.),
         'lh': ({'azimuth': 168.8, 'elevation': 60.3, 
                 'focalpoint': (-0.60, 0., 0.)}, 90, 295.)}

def add_motor_labels(brain, src_df):
    lnames = src_df.index.levels[0].map(lambda l: l[:-7]).unique()
    
    premotor = ss.Glasser_areas[
            ss.Glasser_areas['main section']==8]['area name'].values
    
    for hemi in brain.geo.keys():
        labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_5_8', 
                                            hemi=hemi)
        
        hemilnames = lnames[list(lnames.map(
                lambda s: s[0] == hemi[0].upper()).values)]
        
        for label in labels:
            if label.name in hemilnames:
                if label.name[2:-7] in premotor:
                    col = 'k'
                else:
                    col = 'w'
                brain.add_label(label, borders=True, hemi=hemi, alpha=0.6, 
                                color=col)
                
        for lname, col in zip(['UpperExtremity', 'Ocular'], 
                              ['#225500', '#44aa00']):
            mlabel = mne.read_labels_from_annot(
                    'fsaverage', parc='HCPMMP1_motor', hemi=hemi, regexp=lname)
            brain.add_label(mlabel[0], borders=True, hemi=hemi, alpha=1, 
                            color=col)


#%% plot the data
brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', foreground='k')

add_motor_labels(brain, src_df)

sv.show_label_vertices(src_df, brain, measure, initial_time=400, 
                       **get_colorinfo(measure, src_df))


#%% save figures
#times = [400]
times = src_df.index.levels[1]

outfiles = {}
for time in times:
    brain.set_time(time)
    
    files = {}
    for hemi, view in views.iteritems():
        brain.show_view(*view)
        file = os.path.join(figdir, 'motor_vertices_%s_%s_%s_%d.png' 
                            % (r_name, measure, hemi, time))
        files[hemi] = file
        brain.save_image(file, antialiased=True)
        
    # stitch them together
    outfiles.append(os.path.join(figdir, 'motor_vertices_%s_%s_%d.png' % 
                                 (r_name, measure, time)))
    os.system("montage -tile 2x1 -geometry +0+0 %s %s" % (
        ' '.join([files['lh'], files['rh']]), outfiles[-1]))

if len(times) > 1:
    import shutil
    for ind, file in enumerate(outfiles):
        shutil.copyfile(file, 'tmp_%d.png' % (ind+1))
    
    mvfile = os.path.join(figdir, 'motor_vertices_%s_%s.mp4' % 
                          (r_name, measure))
    os.system('avconv -f image2 -r 2 -i tmp_%%d.png -vcodec mpeg4 -y %s' 
              % mvfile)
    
    os.system('rm tmp_*.png')