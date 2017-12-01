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


#%% selecting the result
# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201707031206.h5'

# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201709061827.h5'

# label mode = mean, baseline (-0.3, 0), dots 1-5, 
# source GLM, dot_x, dot_y, trial regressors
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_singledot_201711131931.h5'

# label mode = mean, baseline (-0.3, 0), response-aligned in [-200, 50]
# source GLM, trial regressors only
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_singledot_201711281123.h5'


#%%
scparams = pd.read_hdf(os.path.join(ss.inf_dir, basefile), 'scalar_params')
response_aligned = ('response_aligned' in scparams.index) and bool(
        scparams.loc['response_aligned'])

if basefile.startswith('source_sequential'):
    regressors = ['dot_x', 'dot_y', 'accev', 'sum_dot_y_prev']

elif basefile.startswith('source_singledot'):
    regressors = ['response']
    
fdr_alpha = 0.01

clusters = ss.get_fdrcorr_clusters(basefile, regressors, fdr_alpha, 
                                   use_basefile=True)


#%% previous multiple comparison corrected clusters based on first finding 
#  temporal clusters and then FDR-correcting their estimated pvals
#measure = 'mlog10p'
#alpha = 0.05
#
#regressors = ['dot_x', 'dot_y', 'abs_dot_x', 'abs_dot_y', 'accev', 'sum_dot_y_prev']
#
#threshold, measure_cdf = ss.find_slabs_threshold(
#    basefile, measure, quantile=1-alpha, regressors=regressors, 
#    verbose=1, return_cdf=True)
#
#clusters = ss.get_fdrcorr_clusters(basefile, regressors, 0.001, 
#                                   measure, threshold, measure_cdf)


#%% define how colormap is set
def get_colorinfo(r_name, clusters):
    Nsig = ((  clusters.loc['response'].end_t 
             - clusters.loc['response'].start_t) / 10 + 1).sum()
    
    # if sufficiently many significant effects
    if Nsig >= 12:
        # set non-significant effects to NaN
        src_df_masked = ss.load_src_df(basefile, r_name_cmap, clusters, 
                                       use_basefile)
    else:
        # there are not sufficiently many significant effects after FDR, 
        # so don't mask
        src_df_masked = ss.load_src_df(basefile, r_name_cmap, None, 
                                       use_basefile)
    
    if show_measure not in src_df_masked.columns:
        ss.add_measure(src_df_masked, show_measure)
        
    if Nsig >= 12:
        # make colormap based on distribution of significant effects
        colorinfo = {'fmin': src_df_masked[show_measure].abs().min(),
                     'fmid': src_df_masked[show_measure].abs().median(), 
                     'fmax': src_df_masked[show_measure].abs().max(), 
                     'transparent': True,
                     'colormap': 'auto'}
    else:
        # make colormap based on distribution of all effects
        colorinfo = {
                'fmin': src_df_masked[show_measure].abs().quantile(0.95),
                'fmid': src_df_masked[show_measure].abs().quantile(0.99),
                'fmax': src_df_masked[show_measure].abs().quantile(0.999),
                'transparent': True,
                'colormap': 'auto'}
    
    return colorinfo


#%% load specific regressors
show_measure = 'absmean'
r_name = 'response'

# set 'mask = clusters' to only show significant effects, otherwise set to None
mask = clusters

use_basefile = show_measure in ss.basefile_measures    

if show_measure == 'consistency':
    colorinfo = {'fmin': 0, 'fmid': 0.5, 'fmax': 1, 'transparent': False,
                 'colormap': 'coolwarm'}
    
    x_times = [400]
else:
    if basefile.startswith('source_sequential'):
        # regressor which should be used to define the colormap
        r_name_cmap = 'dot_x'
        
    elif basefile.startswith('source_singledot'):
        r_name_cmap = 'response'
        
    colorinfo = get_colorinfo(r_name_cmap, clusters)
    
    if r_name == 'dot_x':
        x_times = [120, 170, 320, 400]
    elif r_name == 'accev':
        x_times = [20, 150, 230, 300]
        
#        colorinfo['fmin'] = 0
        colorinfo['center'] = 0
    elif r_name == 'dot_y':
        x_times = [120, 170, 320, 400]
    elif r_name == 'response':
        if response_aligned:
            x_times = [-30, 0, 30, 50]
        else:
            x_times = [780, 820, 890]

# load the selected regressors and mask as desired
src_df_masked = pd.concat(
        [ss.load_src_df(basefile, reg, mask, use_basefile) 
         for reg in regressors],
        keys=regressors, names=['regressor', 'label', 'time'])

times = src_df_masked.index.levels[2]

if show_measure not in src_df_masked.columns:
    ss.add_measure(src_df_masked, show_measure)

if basefile.startswith('source_sequential'):
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

hemi = 'rh'

if mask is None:
    filepat_base = 'brain_unmasked_' + show_measure
else:
    filepat_base = 'brain_' + show_measure


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
    
    filepat = os.path.join(figdir, filepat_base + '_{}_{}_%02d.png'.format(
            hemi, r_name))
    
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
    effects = effects[effects[show_measure].abs() > colorinfo['fmin']]
    effects.reset_index(level='label', inplace=True)
    effects.reset_index(drop=True, inplace=True)
    effects['region'] = list(map(ss.get_Glasser_section, effects.label))
    
    return effects.sort_values(show_measure)


#%% stitch images from different hemispheres together
infiles = []
for time in x_times:
    time_idx = times.get_loc(time)
    infiles += [
            os.path.join(figdir, 
                         filepat_base + '_{}_{}_{:02d}.png'.format(
                                 'lh', r_name, time_idx)),
            os.path.join(figdir, 
                         filepat_base + '_{}_{}_{:02d}.png'.format(
                                 'rh', r_name, time_idx))]
outfile = os.path.join(figdir, filepat_base + '_{}.png'.format(r_name))
os.system("montage -tile 2x{:d} -geometry +0+0 {} {}".format(
        len(x_times), ' '.join(infiles), outfile))
