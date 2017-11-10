#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:33 2017

@author: bitzer
"""

from __future__ import print_function
import source_visualisations as sv
import source_statistics as ss
import re
import os
import mne

from surfer import Brain

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% load data
r_name = 'response'
measure = 'tval'
make_figures = True

# can be:
# '' for all first-dot-onset-aligned data
# 'exclate' for first-dot-onset-aligned data excluding data close to response
# 'ralign' for response aligned data
datatype = 'ralign'

is_singledot_regressor = lambda name: (
           (name in ['entropy', 'trial_time', 'response'])
        or (re.match(r'.*_\d+$', name) is not None))

basefile = None
if is_singledot_regressor(r_name):
    if datatype == 'ralign':
        # vertices of pre-motor and motor areas, baseline (-0.3, 0), dots 5-7, 
        # response-aligned in [-200, 50], source GLM, 
        # subject-specific normalisation of DM without centering and scaling by std
        # label_tc normalised across trials, times and subjects
        basefile = 'source_singledot_201711081136.h5'
    elif datatype == '':
        # vertices of pre-motor and motor areas, baseline (-0.3, 0), dots 5-7, 
        # time window 0.7-0.89, source GLM, 
        # subject-specific normalisation of DM without centering and scaling by std
        # label_tc normalised across trials, times and subjects
        basefile = 'source_singledot_201711061631.h5'    
else:
    if datatype == 'exclate':
        # vertices of pre-motor and motor areas, baseline (-0.3, 0), first 5 dots, 
        # data points later than 500 ms before response excluded
        # trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
        # subject-specific normalisation of DM without centering and scaling by std
        # label_tc normalised across trials, times and subjects
        basefile = 'source_sequential_201711031951.h5'
    elif datatype == '':
        # vertices of pre-motor and motor areas, baseline (-0.3, 0), first 5 dots, 
        # trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
        # subject-specific normalisation of DM without centering and scaling by std
        # label_tc normalised across trials, times and subjects
        basefile = 'source_sequential_201710231824.h5'

if len(datatype) > 0 and not datatype.startswith('_'):
    datatype = '_' + datatype

src_df = ss.load_src_df(basefile, r_name, use_basefile=True)
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

sv.show_label_vertices(src_df, brain, measure, 
                       **get_colorinfo(measure, src_df))


#%% save figures
if make_figures:
    #times = [400]
    times = src_df.index.levels[1]
    
    outfiles = []
    for time in times:
        brain.set_time(time)
        
        files = {}
        for hemi, view in views.iteritems():
            brain.show_view(*view)
            file = os.path.join(figdir, 'motor_vertices_%s_%s_%s_%d%s.png' 
                                % (r_name, measure, hemi, time, datatype))
            files[hemi] = file
            brain.save_image(file, antialiased=True)
            
        # stitch them together
        outfiles.append(os.path.join(figdir, 'motor_vertices_%s_%s_%d%s.png' % 
                                     (r_name, measure, time, datatype)))
        os.system("montage -tile 2x1 -geometry +0+0 %s %s" % (
            ' '.join([files['lh'], files['rh']]), outfiles[-1]))
    
    if len(times) > 1:
        import shutil
        for ind, file in enumerate(outfiles):
            shutil.copyfile(file, 'tmp_%d.png' % (ind+1))
        
        mvfile = os.path.join(figdir, 'motor_vertices_%s_%s%s.mp4' % 
                              (r_name, measure, datatype))
        os.system('avconv -f image2 -r 2 -i tmp_%%d.png -vcodec mpeg4 -y %s' 
                  % mvfile)
        
        os.system('rm tmp_*.png')