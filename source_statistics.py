#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:28:44 2017

@author: bitzer
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import itertools
import os
import seaborn as sns
from scipy.interpolate import interp1d
import mne

if os.name == 'posix':
    subjects_dir = 'mne_subjects'
else:
    subjects_dir = os.path.expanduser('~\\BeeMEG\\mne_subjects')
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')

Glasser_sections = pd.read_csv('Glasser2016_sections.csv', index_col=0)
Glasser_areas = pd.read_csv('Glasser2016_areas.csv', index_col=0)


def get_Glasser_section(area):
    if area.startswith('??'):
        section = 'whole hemisphere'
    else:
        section = Glasser_areas[Glasser_areas['area name'] == area[2:-7]]['main section']
        section = Glasser_sections.loc[section].name.values[0]
    
    return section


def find_slabs_threshold(basefile, measure='mu_p_large', quantile=0.99, 
                         bemdir=bem_dir, regressors=None, 
                         exclude=None, verbose=2, return_cdf=False):
    """Finds a value threshold for a measure corresponding to a given quantile.
    
        Pools values of the measure across all fitted regressors. Then finds 
        the value corresponding to the given quantile.
    """
    
    if regressors is None:
        try:
            with pd.HDFStore(os.path.join(bemdir, basefile), 'r') as store:
                regressors = store.first_level_src.columns.levels[1]
        except (FileNotFoundError, OSError):
            with pd.HDFStore(os.path.join('data/inf_results', basefile), 'r') as store:
                regressors = store.first_level.columns.levels[2]
    
    if exclude:
        if exclude == 'trialregs':
            exclude = ['intercept', 'entropy', 'response', 'trial_time', 'motoprep']
        elif exclude == 'dotregs':
            exclude = ['abs_dot_x', 'abs_dot_y', 'accev', 'accev_cflip', 
                       'accsur_pca', 'dot_x', 'dot_x_cflip', 'dot_y', 'move_dist',
                       'sum_dot_y_prev']
        if verbose:
            print('excluding:')
            print('\n'.join(exclude), end='\n\n')
        
        regressors = np.setdiff1d(regressors, exclude)
    
    values = np.array([], dtype=float)
    for r_name in regressors:
        if verbose:
            print('adding ' + r_name)
        
        fname = basefile[:-3] + '_slabs_' + r_name + '.h5'
        
        with pd.HDFStore(os.path.join(bemdir, fname), 'r') as store:
            values = np.r_[values, store.second_level_src[measure].values]
    
    if verbose:
        print('N = %d' % values.size)
    if verbose > 1:
        fig, ax = sns.plt.subplots()
        ax = sns.distplot(values, ax=ax)
    
    qval = np.percentile(values, quantile * 100)
    
    if verbose > 1:
        ax.plot(qval*np.r_[1, 1], ax.get_ylim())
    
    if return_cdf:
        values.sort()
        cdf = interp1d(values, np.arange(values.size) / float(values.size),
                       assume_sorted=True)
        return qval, cdf
    else:
        return qval
    
    
def get_time_clusters(measure_series, threshold, cdf):
    
    clusters = []
    
    for label in measure_series.index.levels[0]:
        label_tc = measure_series.xs(label, level='label')
        
        start = 0
        for key, group in itertools.groupby(label_tc >= threshold):
            length = sum(1 for _ in group)
            if key == True:
                logps = np.log10(1 - cdf(label_tc.values[start : start+length]))
                clusters.append([label, label_tc.index[start], 
                                 label_tc.index[start+length-1], logps.sum()])
                
            start += length
    
    return pd.DataFrame(clusters, 
                        columns=['label', 'start_t', 'end_t', 'log10p'])


def get_fdrcorr_clusters(basefile, regressors, measure, threshold, cdf, 
                         fdr_alpha=0.001):
    clusters = []
    for r_name in regressors:
        srcfile = basefile[:-3] + '_slabs_%s.h5' % r_name
        file = os.path.join(bem_dir, srcfile)
        src_df = pd.read_hdf(file, 'second_level_src')
        clusters.append(get_time_clusters(src_df[measure], threshold, cdf))
    
    clusters = pd.concat(clusters, keys=regressors, 
                         names=['regressor', 'cluster'])
    
    reject, pval = mne.stats.fdr_correction(10**clusters.log10p, fdr_alpha)
    clusters['pval_corrected'] = pval
    clusters = clusters[reject]
    
    try:
        clusters['region'] = clusters.label.apply(get_Glasser_section)
    except IndexError:
        print('No brain regions found for area, skipping that step.')
    
    return clusters
    

def apply_cluster_mask(src_df, clusters, regressor):
    """Set values in src_df to nan which do not belong to a cluster."""
    mask = np.ones(src_df.index.size, dtype=bool)
    
    # remove clusters from mask
    for cl in clusters.xs(regressor, level='regressor').itertuples():
        ind = src_df.index.get_locs((cl.label, slice(cl.start_t, cl.end_t)))
        mask[ind] = False
        
    src_df[mask] = np.nan
    

def load_src_df(basefile, regressor, clusters=None):
    """Load src_df for given regressor and result and mask by clusters, if given."""
    file = os.path.join(bem_dir, basefile[:-3] + '_slabs_%s.h5' % regressor)
    src_df = pd.read_hdf(file, 'second_level_src')
    
    if clusters is not None:
        apply_cluster_mask(src_df, clusters, regressor)
    
    return src_df


def null_inconsistent(values, v_threshold, s_threshold=3):
    """Sets all values to 0 that do not belong to a sequence of length 
    s_threshold of values above v_threshold."""
    mask = np.zeros_like(values, dtype=bool)
    
    start = 0
    for key, group in itertools.groupby(values >= v_threshold):
        length = sum(1 for _ in group)
        if key == True and length >= s_threshold:
            mask[start : start+length] = True
        start += length
    
    values = values.copy()
    
    # if the value threshold is below 0, then values were log-transformed
    # so that 0 becomes -inf
    if v_threshold < 0:
        values[~mask] = -np.inf
    else:
        values[~mask] = 0
    
    return values


def null_inconsistent_measure(vseries, v_threshold, s_threshold=3):
    """Applies null_inconsistent to all labels in the series, but not across 
    labels."""
    vseries = vseries.reset_index(level='label')
    vdf = vseries.pivot(columns='label')
    
    vdf = vdf.apply(lambda x: null_inconsistent(x, v_threshold, s_threshold))
    vdf = vdf.stack().reorder_levels(['label', 'time']).sort_index()
    
    return vdf[vdf.columns[0]]


def get_significant_areas_in_time(measure_series, threshold, timewindow):
    values = null_inconsistent_measure(measure_series, threshold)
    values_tmp = values.loc[(slice(None), slice(*timewindow))]
    values_tmp = values_tmp.reset_index(level='label').pivot(columns='label')
    labels = values_tmp.columns.get_level_values('label')[values_tmp.sum() > 0]
    del values_tmp
    
    areas = pd.Series(np.unique(labels.map(lambda l: l[2:-7])))
    
    area_values = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], areas], 
                                       names = ['hemi', 'region']))

    for hemi in ['l', 'r']:
        # create full label names
        labels = areas.map(
                lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
        
        # set values of all areas to that of aggregated value
        avals = values.loc[(labels, slice(None))]
        avals = avals.reset_index(level='time').pivot(columns='time')
        area_values.loc[:, (hemi.upper(), slice(None))] = avals.values.T

    return area_values.drop(area_values.columns[area_values.sum() == 0], axis=1)


def get_significant_areas_in_region(measure_series, threshold, region):
    areas = Glasser_areas[Glasser_areas['main section'] == region]
    areas = areas.sort_values('area name')
    
    values = null_inconsistent_measure(measure_series, threshold)
    
    area_values = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], areas['area name']], 
                                       names = ['hemi', 'region']))

    for hemi in ['l', 'r']:
        # create full label names
        labels = areas['area name'].map(
                lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
        
        # set values of all areas to that of aggregated value
        avals = values.loc[(labels, slice(None))]
        avals = avals.reset_index(level='time').pivot(columns='time')
        area_values.loc[:, (hemi.upper(), slice(None))] = avals.values.T

    return area_values.drop(area_values.columns[area_values.sum() == 0], axis=1)


def get_significant_regions(measure_series, threshold, 
                            region_aggfun=lambda a: np.max(a, axis=0)):
    values = null_inconsistent_measure(measure_series, threshold)
    
    regions = pd.DataFrame(
            [], dtype=float, index=values.index.levels[1], columns=
            pd.MultiIndex.from_product([['L', 'R'], Glasser_sections.name], 
                                       names = ['hemi', 'region']))
    for section in Glasser_sections.index:
        # find all areas in region
        areas = Glasser_areas[Glasser_areas['main section'] == section]
        
        for hemi in ['l', 'r']:
            # create full label names
            labels = areas['area name'].map(
                    lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
            
            # set values of all areas to that of aggregated value
            avals = values.loc[(labels, slice(None))]
            avals = avals.reset_index(level='time').pivot(columns='time')
            regions.loc[:, (hemi.upper(), Glasser_sections.loc[section, 'name'])] = (
                    region_aggfun(avals).values)
    
    return regions.drop(regions.columns[regions.sum() == 0], axis=1)