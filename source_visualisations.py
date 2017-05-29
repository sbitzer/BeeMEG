#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:46 2017

to interact with Mayavi's mlab call ipython using (requires wxpython):

    ETS_TOOLKIT=wx ipython --gui=wx

probably QT is better, but I couldn't get it to work properly
see http://docs.enthought.com/mayavi/mayavi/mlab.html#simple-scripting-with-mlab
for further information about the pitfalls in trying to run mlab interactively

@author: bitzer
"""

import os
import re
import mne
import numpy as np
import glob
import pandas as pd
from surfer import Brain
import matplotlib
import seaborn as sns
import itertools


if os.name == 'posix':
    subjects_dir = 'mne_subjects'
else:
    subjects_dir = os.path.expanduser('~\\BeeMEG\\mne_subjects')
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')
fig_dir = os.path.join(subjects_dir, subject, 'figures')

resultid_re = re.compile(r'.*_(\d{12})_.*.stc')

view = {'lh': {'azimuth': -55.936889415680255,
               'elevation': 28.757539951598851,
               'distance': 441.80865427261426,
               'focalpoint': np.r_[1.97380947, -20.13409592, -19.44953706]}}

Glasser_sections = pd.read_csv('Glasser2016_sections.csv', index_col=0)
Glasser_areas = pd.read_csv('Glasser2016_areas.csv', index_col=0)


def set_regions(vseries, aggfun=lambda a: np.max(a, axis=0)):
    """Set all vertices belonging to a Glasser region/section to one value.
    
        Based on a given aggregation function working on areas in the region.
        vseries must be a pandas Series with label names (areas) in the index.
    """
    for section in Glasser_sections.index:
        # find all areas in region
        areas = Glasser_areas[Glasser_areas['main section'] == section]
        
        for hemi in ['l', 'r']:
            # create full label names
            labels = areas['area name'].map(
                    lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
            
            # set values of all areas to that of aggregated value
            values = vseries.loc[(labels, slice(None))]
            values = values.reset_index(level='time').pivot(columns='time')
            vseries.loc[(labels, slice(None))] = np.tile(
                    aggfun(values).values, len(labels))
            
    return vseries


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


def find_slabs_threshold(srcfile, measure='mu_p_large', quantile=0.99, 
                         bemdir=bem_dir, verbose=2):
    """Finds a value threshold for a measure corresponding to a given quantile.
    
        Pools values of the measure across all fitted regressors. Then finds 
        the value corresponding to the given quantile.
    """
    with pd.HDFStore(os.path.join(bemdir, srcfile), 'r') as store:
        regressors = store.first_level_src.columns.levels[1]
    
    values = np.array([], dtype=float)
    for r_name in regressors:
        if r_name == 'intercept':
            if verbose:
                print('skipping intercept')
            continue
        
        if verbose:
            print('adding ' + r_name)
        
        fname = srcfile[:-3] + '_slabs_' + r_name + '.h5'
        
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
    
    return qval


def load_source_estimate(r_name='dot_x', f_pattern=''):
    # find files
    files = glob.glob(os.path.join(  bem_dir, '*' + f_pattern 
                                   + r_name + '-lh.stc'))
    
    if len(files) > 0:
        print "using file: " + files[0]
        stc = mne.read_source_estimate(files[0])
        stc.resultid = resultid_re.match(files[0]).groups()[0]
        stc.r_name = r_name
    else:
        raise ValueError('No appropriate source file found!')
    
    return stc


def make_movie(stc, hemi='lh', td=10, smoothing_steps=5, colvals=None):
    brain = stc.plot(subject='fsaverage', surface='inflated', hemi=hemi, 
                     smoothing_steps=smoothing_steps)
    brain.show_view(view[hemi])
    
    if colvals is not None:
        brain.scale_data_colormap(*colvals)
    
    brain.save_movie(os.path.join(fig_dir, 'source_'+stc.resultid+'_'
                                           +stc.r_name+'_'+hemi+'.mp4'), 
                                  time_dilation=td)
        
    return brain

    
def time_viewer(stc, hemi='both'):
    brain = stc.plot(subject='fsaverage', surface='inflated', 
                     time_viewer=True, hemi=hemi)
    
    return brain


def make_stc(srcfile, measure, r_name=None, src_df=None, transform=None, 
             mask=None, mask_other=-1):
    file = os.path.join(bem_dir, srcfile)
    
    if srcfile.find('_slabs_') >= 0:
        slabsi = file.find('slabs_')
        r_n = file[slabsi+6:-3]
        if r_name is None:
            r_name = r_n
        elif r_name != r_n:
            raise ValueError("Provided source file name and name of desired "
                             "regressor are not compatible!")
        
        if src_df is None:
            src_df = pd.read_hdf(file, 'second_level_src')
            
        data = src_df[measure].reset_index(level='time')
        data = data.pivot(columns='time')
        if transform is not None:
            data = transform(data)

        if mask is not None:
            data.where(mask, other=mask_other, inplace=True)

        stc = mne.SourceEstimate(
            data.values, 
            vertices=[data.loc['lh'].index.values, 
                      data.loc['rh'].index.values],
            tmin=data.columns.levels[1][0] / 1000.,
            tstep=np.diff(data.columns.levels[1][:2])[0] / 1000.,
            subject='fsaverage')
        
        stc.r_name = r_name
        stc.resultid = measure
        
    return stc


def show_labels_as_data(src_df, measure, brain, labels=None, transform=None,
                        parc='HCPMMP1', transparent=None, colormap='auto',
                        initial_time=None, colorbar=True, clim='auto',
                        threshold=0, region_aggfun=None):
    """Uses efficient brain.add_data to show activation of Glasser areas/labels.
    
        Set threshold > 0 to only show values that are above threshold for some
        time steps (3 by default, cf. null_inconsistent).
        
        Set region_aggfun to some suitable function to aggregate Glasser 
        areas/labels into regions/sections. 
    """
    
    if type(src_df) == str:
        parc = re.match('source_(\w+)_allsubs_\d+_slabs_\w+.h5', srcfile).group(1)
        
        file = os.path.join(bem_dir, srcfile)
        src_df = pd.read_hdf(file, 'second_level_src')
    
    T = src_df.index.levels[1].size
    values = src_df[measure]
    
    # remove 'insignificant' data
    if threshold > 0:
        values = null_inconsistent_measure(values, threshold)
    # aggregate areas into regions
    if region_aggfun is not None:
        values = set_regions(values, region_aggfun)
    
    if labels is None:
        # assume that no labels shown in brain, yet
#        brain.add_annotation(parc)
        
        labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both')
        labels = {l.name: l for l in labels}
        
    if len(labels) != src_df.index.levels[0].size:
        raise ValueError('Number of labels ({}) does not fit to number of '
                         'sources in data ({})!'.format(len(labels),
                         src_df.index.levels[0].size))

    # use MNE color stuff
    ctrl_pts, colormap = mne.viz._3d._limits_to_control_points(
            clim, src_df[measure], colormap)
    if colormap in ('mne', 'mne_analyze'):
        colormap = mne.viz.utils.mne_analyze_colormap(ctrl_pts)
        scale_pts = [-1 * ctrl_pts[-1], 0, ctrl_pts[-1]]
        transparent = False if transparent is None else transparent
    else:
        scale_pts = ctrl_pts
        transparent = True if transparent is None else transparent

    for hemi in ['lh', 'rh']:
        # create data from source estimates
        V = brain.geo[hemi].x.size
        data = np.zeros((V, T))
        for name in [l for l in labels.keys() if l.endswith(hemi)]:
            data[labels[name].vertices, :] = values.xs(name, level='label')
        
        # plot
        brain.add_data(data, time=src_df.index.levels[1].values, 
                       time_label=lambda x: '%4d ms' % (x * 1000),
                       colormap=colormap, colorbar=colorbar, hemi=hemi, 
                       remove_existing=True)
        
    # scale colormap
    if threshold > 0:
        scale_pts[0] = threshold
    brain.scale_data_colormap(fmin=scale_pts[0], fmid=scale_pts[1],
                              fmax=scale_pts[2], transparent=transparent)
    
    # set time (index) to display
    if initial_time is not None:
        brain.set_time(initial_time)
        
    return labels


def show_labels(srcfile, measure, brain=None, r_name=None, src_df=None, 
                transform=None, surface='white', time=0.17, labels=None):
    parc = re.match('source_(\w+)_allsubs_\d+_slabs_\w+.h5', srcfile).group(1)
    if labels is None:
        labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both')
        labels = {l.name: l for l in labels}
    
    file = os.path.join(bem_dir, srcfile)
    
    if srcfile.find('_slabs_') >= 0:
        slabsi = file.find('slabs_')
        r_n = file[slabsi+6:-3]
        if r_name is None:
            r_name = r_n
        elif r_name != r_n:
            raise ValueError("Provided source file name and name of desired "
                             "regressor are not compatible!")
        
    if src_df is None:
        src_df = pd.read_hdf(file, 'second_level_src')
        
    if brain is None:
        brain = Brain('fsaverage', 'both', surface, cortex='low_contrast',
                      subjects_dir=subjects_dir)
    else:
        brain.remove_labels(hemi='rh')
        brain.remove_labels(hemi='lh')
    
    data = src_df.loc[(slice(None), time), measure]
    
    if transform is not None:
        data = transform(data)

    vmin = 0.9
    
    norm = matplotlib.colors.Normalize(vmin, 1, clip=True)
    alpha = matplotlib.colors.Normalize(vmin, (1-vmin)/2+vmin, clip=True)
    
    data = data[data > vmin]
    
    cmap = matplotlib.cm.hot
    
    print data
    
    for index, value in data.iteritems():
        if index[0].startswith('L'):
            hemi = 'left'
        else:
            hemi = 'right'
        
        print index[0]
        brain.add_label(labels[index[0]], color=cmap(norm(value)), 
                        alpha=alpha(value), hemi=hemi)
        
    return brain


if __name__ == '__main__':
    masked_hot = np.r_[np.zeros((128, 4)), 
                       matplotlib.cm.hot(np.linspace(0, 1, 128))] * 255
    
    srcfile = 'source_allsubs_201703301614_slabs_accev.h5'
    src_df = pd.read_hdf(os.path.join(bem_dir, srcfile), 'second_level_src')
    stc_mupl = make_stc(srcfile, 'mu_p_large', src_df=src_df)
    stc_mut = make_stc(srcfile, 'mu_t', src_df=src_df)
    mask = stc_mupl.data > 0.95
#    mask = stc_mut.data > 3.5
    
    stc = make_stc(srcfile, 'consistency', src_df=src_df, mask=mask)
    
    brain = stc.plot(subject='fsaverage', surface='inflated', hemi='both',
                     colormap=masked_hot, transparent=False, 
                     clim={'kind': 'value', 'lims': [-1, 0, 1]},
                     initial_time=0.35, smoothing_steps=5)
