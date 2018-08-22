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
import mne

from surfer import Brain

figdir = sv.fig_dir


#%%
# settings common across all regressors
fdr_alpha = 0.05

common_color_scale = True

# newest results are based on this parcellation, cf. 'srcfile' in basefile
parc = 'HCPMMP1_5_8'

show_measure = 'abstval'
use_basefile = show_measure in ss.basefile_measures

# regressor of interest
r_name = 'dot_x_sign'

# settings for specific regressors (can overwrite common settings)
if r_name == 'dot_x':
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
    #basefile = 'source_sequential_201801291241.h5'
    
    # loose source orientations, but cortex normal currents
    # label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
    # time window [0, 690], exclude time-outs, local normalisation of DM
    # trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
    # label_tc normalised across trials but within times and subjects
    basefile = 'source_sequential_201807041930.h5'
    
    twins = {"early": [110, 130], 
             "transition": [160, 200], 
             "plateau": [300, 500]}
    
elif r_name == 'sum_dot_x':
    # loose source orientations, but cortex normal currents
    # label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
    # time window [0, 690], exclude time-outs, local normalisation of DM
    # trialregs_dot=0, sum_dot_x instead of dot_x and accev, sum_dot_y_prev, 
    # percupt, constregs=0 for 1st dot, 
    # label_tc normalised across trials but within times and subjects
    basefile = 'source_sequential_201808131739.h5'
    
    twins = {"full": [0, 550]}
    
elif r_name == 'dot_y':
    # loose source orientations, but cortex normal currents
    # label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
    # time window [0, 690], exclude time-outs, local normalisation of DM
    # trialregs_dot=0, sum_dot_x instead of dot_x and accev, sum_dot_y_prev, 
    # percupt, constregs=0 for 1st dot, 
    # label_tc normalised across trials but within times and subjects
    basefile = 'source_sequential_201808131739.h5'
    
    twins = {"one": [120, 220],
             "two": [300, 400]}

elif r_name == 'dot_x_sign':
    # loose source orientations, but cortex normal currents
    # label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
    # time window [0, 690], exclude time-outs, local normalisation of DM
    # trialregs_dot=0, dot_x_sign, accev_sign, sum_dot_y_prev, 
    # percupt, constregs=0 for 1st dot, 
    # label_tc normalised across trials but within times and subjects
    basefile = 'source_sequential_201808161156.h5'
    
    twins = {"early": [110, 130], 
             "transition": [160, 200], 
             "plateau": [300, 500]}
    


#%% load clusters of significant effects
clusters = ss.get_fdrcorr_clusters(basefile, [r_name], fdr_alpha, 
                                   use_basefile=True)


#%% save all significant r_name clusters to csv-file
def to_csv(areas, fname):
    areas['label'] = areas['label'].map(lambda x: x[:-7])
    areas.to_csv(fname)
    
to_csv(clusters.loc[r_name].copy().sort_values('start_t')[
        ['label', 'region', 'start_t', 'end_t', 'log10p']],
       os.path.join(figdir, 'significant_clusters_{}.csv'.format(r_name)))


#%%
src_df = ss.load_src_df(basefile, r_name, None, use_basefile)

if show_measure not in src_df.columns:
    ss.add_measure(src_df, show_measure)

labels = src_df.index.levels[0]


def get_average_effects(twin, wname, top=5):
    winclusters = clusters.loc[r_name]
    winclusters = winclusters[
             ((winclusters.start_t >= twin[0]) 
            & (winclusters.start_t <= twin[1])) 
            | ((winclusters.end_t >= twin[0]) 
            & (winclusters.end_t <= twin[1]))]
    areas = winclusters.label.unique()
    
    time = int(np.mean(twin))
    
    avsrcdf = pd.DataFrame(np.zeros((labels.size, src_df.shape[1])),
                           index=pd.MultiIndex.from_product(
                                   [labels, [time]],
                                   names=['label', 'time']),
                           columns=src_df.columns)
        
    avsrcdf.loc[list(areas), :] = src_df.loc[
            (list(areas), slice(*twin)), :].groupby('label').mean().values
    
    # save computed values in csv
    avcsv = avsrcdf[avsrcdf.tval != 0].copy().xs(time, level='time')
    avcsv['region'] = avcsv.index.map(ss.get_Glasser_section)
    avcsv = avcsv.reset_index().sort_values(show_measure, ascending=False)
    to_csv(avcsv[['label', 'region', 'mlog10p', 'tval', 'mean', 'std']].copy(),
           os.path.join(figdir, 'average_significant_%s_%s.csv' % (
                   r_name, wname)))
    
    # areas with largest effects
    toplabels = {}
    for hemi in ['L', 'R']:
        ind = avsrcdf.index.get_level_values('label').map(
                lambda x: x.startswith(hemi))
        
        avtop = avsrcdf[ind].abs().sort_values(
                show_measure, ascending=False).head(top)
        avtop = avtop[avtop[show_measure] != 0]
        
        toplabels[hemi.lower() + 'h'] = list(
                avtop.xs(time, level='time').index)
    
    return avsrcdf, time, toplabels


#%% compute all results so that you can determine a common colour scale
results = {}
for name, win in twins.items():
    avsrcdf, time, toplabels = get_average_effects(win, name)
    results[name] = dict(avsrcdf=avsrcdf, time=time, toplabels=toplabels)

avsrcdf = pd.concat([results[name]['avsrcdf'] for name in twins.keys()])

def get_colorinfo(avsrcdf):
    avsrcdf = avsrcdf[avsrcdf[show_measure] != 0]
    
    return {'fmin': avsrcdf[show_measure].abs().min(),
            'fmid': avsrcdf[show_measure].abs().median(), 
            'fmax': avsrcdf[show_measure].abs().max(), 
            'transparent': True,
            'colormap': 'auto'}

        
#%% make helper plotting function
filepat_base = 'av_brain_{}_{}'.format(r_name, show_measure)

def brain_plot(brain, wname, toplabels=False, save=False):
    if r_name in ['dot_x', 'dot_y', 'dot_x_sign']:
        views = {'rh': ['medial', 'parietal'], #{'azimuth': -20, 'elevation': 62, 'roll': -68}], 
                 'lh': ['parietal', 'medial']}
    elif r_name == 'sum_dot_x':
        views = {'rh': ['medial', 'lateral'],
                 'lh': ['lateral', 'medial']}
    
    if type(brain) is str:
        hemi = brain
        brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                      subjects_dir=sv.subjects_dir, background='w', 
                      foreground='k')
    else:
        hemi = brain.geo.keys()[0]
        
    if common_color_scale:
        cinfo = get_colorinfo(avsrcdf)
    else:
        cinfo = get_colorinfo(results[wname]['avsrcdf'])
        
    sv.show_labels_as_data(results[wname]['avsrcdf'], show_measure, brain, 
                           time_label=None, parc=parc, **cinfo)
    
    if toplabels:
        brain.remove_labels()
        
        alllabels = mne.read_labels_from_annot(
                'fsaverage', parc=parc, hemi=hemi)
        for label in alllabels:
            if label.name in results[wname]['toplabels'][hemi]:
                brain.add_label(label, borders=1, hemi=hemi, alpha=0.8, 
                                color='k')
    
    # increase font size of colorbar - this only works by increasing the 
    # colorbar itself and setting the ratio of colorbar to text
    brain.data['colorbar'].scalar_bar.bar_ratio = 0.35
    brain.data['colorbar'].scalar_bar_representation.position = [0.075, 0.01]
    brain.data['colorbar'].scalar_bar_representation.position2 = [0.85, 0.12]
    
    if save:
        filepat = os.path.join(figdir, filepat_base + '_{}_{}.png'.format(
                wname, hemi))
        
        brain.save_montage(filepat, views[hemi], colorbar=0)
    
    return brain


#%%
interactive = False
show_toplabels = True

if interactive:
    hemi = 'rh'
    wname = 'full'
    brain = brain_plot(hemi, wname, show_toplabels)
else:
    for hemi in ['lh', 'rh']:
        first = True
        for wname in twins.keys():
            if first:
                brain = brain_plot(hemi, wname, show_toplabels, True)
            else:
                brain_plot(brain, wname, show_toplabels, True)
            
        brain.close()


#%% stitch images from different hemispheres together
if not interactive:
    for wname in twins.keys():
        infiles = [
                os.path.join(figdir, 
                             filepat_base + '_{}_{}.png'.format(wname, hemi))
                for hemi in ['lh', 'rh']]
        outfile = os.path.join(figdir, filepat_base + '_{}.png'.format(wname))
        os.system("montage -tile 2x1 -geometry +0+0 {} {}".format(
                ' '.join(infiles), outfile))
