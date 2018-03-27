#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:28:21 2018

@author: bitzer
"""

from __future__ import print_function
import source_visualisations as sv
import source_statistics as ss
import numpy as np
import pandas as pd
import os

from surfer import Brain

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%%
# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201709061827.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201801261754.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
basefile = 'source_sequential_201801291241.h5'

if basefile.startswith('source_sequential'):
    regressors = ['dot_x', 'dot_y', 'accev', 'sum_dot_y_prev']

elif basefile.startswith('source_singledot'):
    regressors = ['response']
    
fdr_alpha = 0.01

clusters = ss.get_fdrcorr_clusters(basefile, regressors, fdr_alpha, 
                                   use_basefile=True)

# save all significant dot_x clusters to csv-file
def to_csv(areas, fname):
    areas['label'] = areas['label'].map(lambda x: x[:-7])
    areas.to_csv(fname)
    
to_csv(clusters.loc['dot_x'].copy().sort_values('start_t')[
        ['label', 'region', 'start_t', 'end_t', 'log10p']],
       os.path.join(figdir, 'significant_clusters.csv'))


#%% find those areas which have at least one significant cluster within time
# period of interest
show_measure = 'tval'
use_basefile = show_measure in ss.basefile_measures

r_name = 'dot_x'

twin = [300, 500]

winclusters = clusters.loc[r_name]
winclusters = winclusters[
         ((winclusters.start_t >= twin[0]) 
        & (winclusters.start_t <= twin[1])) 
        | ((winclusters.end_t >= twin[0]) 
        & (winclusters.end_t <= twin[1]))]

areas = winclusters.label.unique()


#%%
src_df = ss.load_src_df(basefile, r_name, None, use_basefile)

if show_measure not in src_df.columns:
    ss.add_measure(src_df, show_measure)

labels = src_df.index.levels[0]
avsrcdf = pd.DataFrame(np.zeros((labels.size, src_df.shape[1])),
                       index=pd.MultiIndex.from_product(
                               [labels, [int(np.mean(twin))]],
                               names=['label', 'time']),
                       columns=src_df.columns)
    
avsrcdf.loc[list(areas), :] = src_df.loc[
        (list(areas), slice(*twin)), :].groupby('label').mean().values

times = avsrcdf.index.levels[1]

avcsv = avsrcdf[avsrcdf.tval != 0].copy().xs(int(np.mean(twin)), level='time')
avcsv['region'] = avcsv.index.map(ss.get_Glasser_section)
to_csv(avcsv.reset_index()[
        ['label', 'region', 'mlog10p', 'tval', 'mean', 'std']].sort_values(
        'mlog10p', ascending=False),
       os.path.join(figdir, 'average_significant_%d-%d.csv' % (
               twin[0], twin[1])))


#%% print areas with largest effects
for hemi in ['L', 'R']:
    ind = avsrcdf.index.get_level_values('label').map(
            lambda x: x.startswith(hemi))
    
    print('\nhemi: ' + hemi)
    print(avsrcdf[ind].abs().sort_values('tval', ascending=False).head(5))


#%% 
colorinfo = {'fmin': avsrcdf.loc[list(areas), show_measure].abs().min(),
             'fmid': avsrcdf.loc[list(areas), show_measure].abs().median(), 
             'fmax': avsrcdf.loc[list(areas), show_measure].abs().max(), 
             'transparent': True,
             'center': 0,
             'colormap': 'auto'}

        
#%% make helper plotting function
filepat_base = 'av_brain_' + show_measure

def brain_plot(brain):
    views = {'rh': ['medial', 'lateral'], #{'azimuth': -20, 'elevation': 62, 'roll': -68}], 
             'lh': ['lateral', 'medial']}
    
    if type(brain) is str:
        hemi = brain
        brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                      subjects_dir=sv.subjects_dir, background='w', 
                      foreground='k')
    else:
        hemi = brain.geo.keys()[0]
        
    sv.show_labels_as_data(avsrcdf, show_measure, 
                           brain, time_label=None, **colorinfo)
    
    # increase font size of colorbar - this only works by increasing the 
    # colorbar itself and setting the ratio of colorbar to text
    brain.data['colorbar'].scalar_bar.bar_ratio = 0.35
    brain.data['colorbar'].scalar_bar_representation.position = [0.075, 0.01]
    brain.data['colorbar'].scalar_bar_representation.position2 = [0.85, 0.12]
    
    filepat = os.path.join(figdir, filepat_base + '_{}_{}.png'.format(
            hemi, r_name))
    
    brain.save_montage(filepat, views[hemi], colorbar=0)
    
    return brain


#%%
hemi = 'lh'
brain = brain_plot(hemi)

#%% stitch images from different hemispheres together
infiles = []
for time in times:
    time_idx = times.get_loc(time)
    infiles += [
            os.path.join(figdir, 
                         filepat_base + '_{}_{}.png'.format(
                                 'lh', r_name)),
            os.path.join(figdir, 
                         filepat_base + '_{}_{}.png'.format(
                                 'rh', r_name))]
outfile = os.path.join(figdir, filepat_base + '_{}.png'.format(r_name))
os.system("montage -tile 2x1 -geometry +0+0 {} {}".format(
        ' '.join(infiles), outfile))
