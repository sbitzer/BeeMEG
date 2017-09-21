#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:53:11 2017

@author: bitzer
"""

import source_visualisations as sv
import source_statistics as ss
import pandas as pd
import os

from surfer import Brain

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% selecting the base measure and result
measure = 'mu_p_large'

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
show_measure = 'mu_mean'
r_name = 'accev'

src_df_masked = pd.concat(
        [ss.load_src_df(basefile, reg, clusters) for reg in regressors],
        keys=regressors, names=['regressor', 'label', 'time'])

times = src_df_masked.index.levels[2]

# flip sign of all non-nan values in accev for which there is a nan value
# in dot_x 100 ms later - these are the effects that cannot be explained by the
# correlation of accev with the dot_x of the previous dot
fliptimes = times[times <= times[-1]-100]
accevvals = src_df_masked.loc[('accev', slice(None), fliptimes), show_measure]
dotxvals = src_df_masked.loc[('dot_x', slice(None), fliptimes+100), show_measure]
flip = pd.np.ones(dotxvals.size)
flip[dotxvals.isnull().values] = -1
src_df_masked.loc[('accev', slice(None), fliptimes), show_measure] = accevvals * flip

if show_measure == 'consistency':
    colorinfo = [0, 0.5, 1, False]
    colormap = 'coolwarm'
    clim = {'kind': 'value', 'lims': [0, 0.5, 1]}
    
    x_times = [400]
else:
    if r_name == 'dot_x':
        colorinfo = [src_df_masked.loc[r_name, show_measure].min(),
                     src_df_masked.loc[r_name, show_measure].median(), 
                     src_df_masked.loc[r_name, show_measure].max(), 
                     True]
        colormap = 'auto'
        clim = 'auto'
        
        x_times = [120, 170, 320, 400]
    elif r_name == 'accev':
        x_times = [20, 120, 230, 300]
        
        colormap = 'divtrans'
        clim = [0, 
                src_df_masked.loc['dot_x', show_measure].median(), 
                src_df_masked.loc['dot_x', show_measure].max()]

hemi = 'rh'


#%% make helper plotting function
def brain_plot(r_name, viewtimes, brain, colormap='auto', clim='auto'):
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
                           brain, transparent=True, colormap=colormap, 
                           clim=clim)
    
    if colormap != 'divtrans':
        brain.scale_data_colormap(*colorinfo)
    
    filepat = os.path.join(figdir, 'brain_{}_{}_{}_%02d.png'.format(
            show_measure, hemi, r_name))
    
    brain.save_image_sequence([times.get_loc(t) for t in viewtimes],
                              filepat, montage=views[hemi])
    
    return brain
    

#%% plot effects for selected regressor
brain = brain_plot(r_name, x_times, hemi, colormap=colormap, clim=clim)


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