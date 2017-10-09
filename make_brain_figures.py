#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:53:11 2017

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


#%% selecting the clustering measure and result
measure = 'mlog10p'

# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201707031206.h5'

# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201709061827.h5'

alpha = 0.05


#%% get threshold for dot regressors of interest
regressors = ['dot_x', 'dot_y', 'abs_dot_x', 'abs_dot_y', 'accev', 'sum_dot_y_prev']

threshold, measure_cdf = ss.find_slabs_threshold(
    basefile, measure, quantile=1-alpha, regressors=regressors, 
    verbose=1, return_cdf=True)


#%% identify significant clusters
clusters = ss.get_fdrcorr_clusters(basefile, regressors, measure, threshold, 
                                   measure_cdf, fdr_alpha=0.001)


#%% load specific regressors
regressors = ['dot_x', 'dot_y', 'accev']
show_measure = 'absmean'
r_name = 'dot_x'

use_basefile = show_measure in ss.basefile_measures

src_df_masked = pd.concat(
        [ss.load_src_df(basefile, reg, clusters, use_basefile) 
         for reg in regressors],
        keys=regressors, names=['regressor', 'label', 'time'])

times = src_df_masked.index.levels[2]

if show_measure not in src_df_masked.columns:
    ss.add_measure(src_df_masked, show_measure)

# flip sign of all non-nan values in accev for which there is a nan value
# in dot_x 100 ms later - these are the effects that are only present in 
# accumulated evidence, but not in dot_x
fliptimes = times[times <= times[-1]-100]
accevvals = src_df_masked.loc[('accev', slice(None), fliptimes), show_measure]
dotxvals = src_df_masked.loc[('dot_x', slice(None), fliptimes+100), show_measure]
flip = np.ones(dotxvals.size)
flip[dotxvals.isnull().values] = -1
print('number of flips = %d' % np.sum(flip < 0))
# additionally flip all non-nan values for which the sign of the effect differs
# note that the meaning of accev is such that its sign is flipped with respect
# to dot_x (dot_x is large for evidence for right while accev measures evidence
# for left), this means that you should only those effects are different whose
# signs are actually equal
signdiff = (
          np.sign(src_df_masked.loc[('accev', slice(None), fliptimes), 
                                    'mean'].values) 
        - np.sign(src_df_masked.loc[('dot_x', slice(None), fliptimes+100), 
                                    'mean'].values))
flip[signdiff == 0] = -1
print('number of flips with signdiff = %d' % np.sum(flip < 0))
src_df_masked.loc[('accev', slice(None), fliptimes), show_measure] = accevvals * flip

if show_measure == 'consistency':
    colorinfo = {'fmin': 0, 'fmid': 0.5, 'fmax': 1, 'transparent': False,
                 'colormap': 'coolwarm'}
    
    x_times = [400]
else:
    if r_name == 'dot_x':
        colorinfo = {'fmin': src_df_masked.loc[r_name, show_measure].abs().min(),
                     'fmid': src_df_masked.loc[r_name, show_measure].abs().median(), 
                     'fmax': src_df_masked.loc[r_name, show_measure].abs().max(), 
                     'transparent': True,
                     'colormap': 'auto'}
        
        x_times = [120, 170, 320, 400]
    elif r_name == 'accev':
        x_times = [20, 150, 230, 300]
        
        colorinfo = {'fmin': 0,
                     'fmid': src_df_masked.loc['dot_x', show_measure].abs().median(), 
                     'fmax': src_df_masked.loc['dot_x', show_measure].abs().max(),
                     'transparent': True,
                     'center': 0,
                     'colormap': 'auto'}
        
        colormap = 'auto'

hemi = 'rh'


#%% make helper plotting function
def brain_plot(r_name, viewtimes, brain):
    views = {'rh': ['medial', 'parietal'], 
             'lh': ['parietal', 'medial']}
    
    if type(brain) is str:
        hemi = brain
        brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                      subjects_dir=sv.subjects_dir, background='w', 
                      foreground='k')
    else:
        hemi = brain.geo.keys()[0]
        
    sv.show_labels_as_data(src_df_masked.loc[r_name], show_measure, 
                           brain, **colorinfo)
    
    filepat = os.path.join(figdir, 'brain_{}_{}_{}_%02d.png'.format(
            show_measure, hemi, r_name))
    
    brain.save_image_sequence([times.get_loc(t) for t in viewtimes],
                              filepat, montage=views[hemi])
    
    return brain
    

#%% plot effects for selected regressor
brain = brain_plot(r_name, x_times, hemi)


#%% list the shown values
def list_effects(time):
    effects = pd.DataFrame(
            src_df_masked.loc[(r_name, slice(None), time), show_measure]
            .dropna().sort_values())
    effects.reset_index(level='label', inplace=True)
    effects.reset_index(drop=True, inplace=True)
    effects['region'] = list(map(ss.get_Glasser_section, effects.label))
    
    return effects.sort_values(show_measure)


#%% stitch images from different hemispheres together
infiles = []
for time in x_times:
    infiles += [
            os.path.join(figdir, 
                         'brain_{}_{}_{}_{:02.0f}.png'.format(
                                 show_measure, 'lh', r_name, time/10)),
            os.path.join(figdir, 
                         'brain_{}_{}_{}_{:02.0f}.png'.format(
                                 show_measure, 'rh', r_name, time/10))]
outfile = os.path.join(figdir, 'brain_{}_{}.png'.format(show_measure, r_name))
os.system("montage -tile 2x{:d} -geometry +0+0 {} {}".format(
        len(x_times), ' '.join(infiles), outfile))


#%% get threshold and clusters for trial regressors of interest
#regressors = ['response']
#
#threshold, measure_cdf = ss.find_slabs_threshold(
#    basefile, measure, quantile=1-alpha, regressors=regressors, 
#    verbose=1, return_cdf=True, exclude='dotregs')
#
## identify significant clusters
#clusters = ss.get_fdrcorr_clusters(basefile, regressors, measure, threshold, 
#                                   measure_cdf, fdr_alpha=0.001)
#
#
##%% plot effects for response
#r_name = 'response'
#show_measure = 'consistency'
#t_off = 200
#
#src_df_masked = pd.concat(
#        [ss.load_src_df(basefile, reg, clusters) for reg in regressors],
#        keys=regressors, names=['regressor', 'label', 'time'])
#
#times = src_df_masked.index.levels[2] + t_off
#
## update time to trial time
#src_df_masked.index = pd.MultiIndex.from_product(
#        src_df_masked.index.levels[:2] + [times], 
#        names=['regressor', 'label', 'time'])
#
#if show_measure == 'consistency':
#    colorinfo = [0, 0.5, 1, False]
#    colormap = 'coolwarm'
#    clim = {'kind': 'value', 'lims': [0, 0.5, 1]}
#else:
#    colorinfo = [src_df_masked[show_measure].min(),
#                 src_df_masked[show_measure].median(), 
#                 src_df_masked[show_measure].max(), 
#                 True]
#    colormap = 'auto'
#    clim = 'auto'
#
#r_times = [320, 780, 880]
#
#for hemi in ['lh', 'rh']:
#    brain = brain_plot(r_name, r_times, hemi, colormap=colormap, 
#                       clim=clim)