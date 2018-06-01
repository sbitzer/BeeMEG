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
r_name = 'dot_x'
measure = 'tval'
fdr_alpha = 0.01

# None for no figure and movie generation
# otherwise a list of times for which figures should be generated
# if more than one times is given, a movie will be created from the figures
make_figures = [490]

# can be:
# '' for all first-dot-onset-aligned data
# 'exclate' for first-dot-onset-aligned data excluding data close to response
# 'ralign' for response aligned data
# 'eog' for EOG event regression results
# 'evoked' evoked (average) signals for a selected time window
datatype = 'exclate'

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
#        basefile = 'source_sequential_201711031951.h5'
        
        # vertices of premotor, motor, paracentral and superior parietal
        # no baseline correction, toolate=-200, time window [250, 600]
        # local normalisation of DM, trial normalisation of data
        # x/y coordinates, absolutes, percupt, accev, sum_dot_y_prev
        basefile = 'source_sequential_201803131115.h5'
    elif datatype == '':
        # vertices of pre-motor and motor areas, baseline (-0.3, 0), first 5 dots, 
        # trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
        # subject-specific normalisation of DM without centering and scaling by std
        # label_tc normalised across trials, times and subjects
        basefile = 'source_sequential_201710231824.h5'
    elif datatype == 'eog':
        # eog event regression
        basefile = 'eog_source_201802091232.h5'
    
    elif datatype == 'evoked':
        # evoked sources 300 to 500 ms after response
#        basefile = 'evoked_source_201802091710.h5'
        # evoked sources -1000 to 1000 after response
        basefile = 'evoked_source_201802271701.h5'

if len(datatype) > 0 and not datatype.startswith('_'):
    datatype = '_' + datatype

src_df = ss.load_src_df(basefile, r_name, use_basefile=True)
ss.add_measure(src_df, 'p_fdr')


#%% prepare plotting
def get_colorinfo(measure, src_df, fdr_alpha=0.01, only_significant=False):
    # find measure value that is the first one with p-value equal or smaller 
    # than fdr_alpha
    pdiff = src_df.p_fdr - fdr_alpha
    try:
        fmid = src_df[pdiff <= 0].sort_values('p_fdr')[measure].abs().iloc[-1]
    except IndexError:
        print('No FDR-corrected significant effects!')
        fmid = src_df[measure].abs().quantile(0.95)
    
    fmax = src_df[measure].abs().max()
    
    if only_significant:
        colorinfo = {'fmin': fmid, 
                     'fmid': fmid + (fmax-fmid) * 0.01, 
                     'fmax': fmax,
                     'transparent': True}
    else:
        colorinfo = {'fmin': fmid / 2., 
                     'fmid': fmid, 
                     'fmax': fmax}
    
    if measure == 'tval':
        colorinfo['center'] = 0
        
    return colorinfo

views = {'rh': ({'azimuth': 6.8, 'elevation': 60.3, 
                 'focalpoint': (-0.60, 0., 0.)}, -90, 295.),
         'lh': ({'azimuth': 168.8, 'elevation': 60.3, 
                 'focalpoint': (-0.60, 0., 0.)}, 90, 295.)}

def add_motor_labels(brain, src_df):
    try:
        lnames = src_df.index.levels[0].map(lambda l: l[:-7]).unique()
    except AttributeError:
        lnames = src_df.index.map(lambda l: l[:-7]).unique()
    
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
                       **get_colorinfo(measure, src_df, fdr_alpha, True))


#%% save figures
if make_figures is not None:
    if len(make_figures) == 0:
        times = src_df.index.levels[1]
    else:
        times = make_figures
    
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