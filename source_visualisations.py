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

from __future__ import print_function
import sys
import os
import re
import mne
import numpy as np
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import source_statistics as ss

try:
    from surfer import Brain
except:
    pass

if os.name == 'posix':
    subjects_dir = 'mne_subjects'
else:
    subjects_dir = os.path.expanduser('~\\BeeMEG\\mne_subjects')
subject = 'fsaverage'
bem_dir = os.path.join(subjects_dir, subject, 'bem')
fig_dir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')

resultid_re = re.compile(r'.*_(\d{12})_.*.stc')

view = {'lh': {'azimuth': -55.936889415680255,
               'elevation': 28.757539951598851,
               'distance': 441.80865427261426,
               'focalpoint': np.r_[1.97380947, -20.13409592, -19.44953706]}}

Glasser_sections = pd.read_csv('Glasser2016_sections.csv', index_col=0)
Glasser_areas = pd.read_csv('Glasser2016_areas.csv', index_col=0)

measure_basevals = {'mu_mean': 0, 'mu_std': np.inf, 'mu_t': 0, 'mu_testval': 0,
                    'mu_p_large': 0, 'sigma_mean': np.inf, 'sigma_std': np.inf,
                    'theta_mean': 0.5, 'theta_std': np.inf, 'lp_mean': 0,
                    'lp_std': np.inf, 'overlap': 0, 'consistency': 0.5,
                    'mean': 0, 'tval': 0, 'mlog10p': 0, 'std': 0, 'abstval': 0,
                    'absmean': 0}

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


def show_timecourses(tc_df, ylim, L=3, logy=False):
    labels = tc_df.columns.get_level_values(1).unique()
    cols = matplotlib.cm.Paired(np.linspace(0, 1, 12))
    cols = cols[np.mod(np.arange(len(labels)), 12)]
    cols = {k: v for k, v in zip(labels, cols)}
    
    def update_cols(ax, df, inds):
        lines = ax.get_lines()
        for line, name in zip(lines, df.columns[inds]):
            line.set_color(cols[name])
    
    for hemi in tc_df.columns.levels[0]:
        active = tc_df.loc[:, hemi]
        N = active.shape[1]
        
        if N > 0:
            fig, axes = plt.subplots(int(np.ceil(N/float(L))), 1, 
                                     sharex=True, 
                                     figsize=[8, max(4, N / float(L) * 1.3)])
            
            try:
                for i, ax in enumerate(axes):
                    inds = slice(i*L, (i+1)*L)
                    active.iloc[:, inds].plot.line(ax=ax)
                    update_cols(ax, active, inds)
                    ax.set_ylim(*ylim)
                    if logy:
                        ax.set_yscale('log')
                    ax.legend(loc='best')
                axes[0].set_title('hemisphere: ' + hemi)
                
            except TypeError:
                inds = slice(0, L)
                active.iloc[:, inds].plot.line(ax=axes)
                update_cols(axes, active, inds)
                axes.set_ylim(*ylim)
                if logy:
                    axes.set_yscale('log')
                axes.legend(loc='best')
                axes.set_title('hemisphere: ' + hemi)


def load_source_estimate(r_name='dot_x', f_pattern=''):
    # find files
    files = glob.glob(os.path.join(  bem_dir, '*' + f_pattern 
                                   + r_name + '-lh.stc'))
    
    if len(files) > 0:
        print("using file: " + files[0])
        stc = mne.read_source_estimate(files[0])
        stc.resultid = resultid_re.match(files[0]).groups()[0]
        stc.r_name = r_name
    else:
        raise ValueError('No appropriate source file found!')
    
    return stc


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


def extract_hemi_data(src_series, hemi):
    assert type(src_series) is pd.Series
    
    data = src_series[src_series.index.get_level_values('label').map(
            lambda s: s[0] == hemi[0].upper()).values].copy()
    
    data.index.set_levels(data.index.levels[0].map(lambda x: int(x[-6:])), 
                          'label', inplace=True)
    
    data = data.unstack('time')
    
    data.index.name = 'vertex'
    data.sort_index(inplace=True)
    data.sort_index(axis=1, inplace=True)
    
    return data


def get_label_sign_flips(lnames, parc='HCPMMP1_5_8', combine=False,
                         srcfile=os.path.join(bem_dir, 
                                              'fsaverage-oct-6-src.fif')):
    # load source spaces
    src = mne.read_source_spaces(srcfile)
    
    # load labels
    labels = mne.read_labels_from_annot('fsaverage', parc)
    
    hemind = dict(lh=0, rh=1)
    
    if combine:
        flips = []
        for hemi in ['lh', 'rh']:
            lind = []
            biglabel = None
            for label in labels:
                if (label.name.startswith(hemi[0].upper()) 
                    and label.name in lnames):
                    
                    if biglabel is None:
                        biglabel = label
                    else:
                        biglabel = biglabel + label
                    
                    vertices = np.intersect1d(label.vertices, 
                                              src[hemind[hemi]]['vertno'])
                    lind += [label.name + '_%06d' % vert for vert in vertices]
        
            flips.append(pd.Series(mne.label.label_sign_flip(biglabel, src),
                                   index=lind))
    else:
        flips = []
        for label in labels:
            if label.name in lnames:
                flip = mne.label.label_sign_flip(label, src)
                
                vertices = np.intersect1d(label.vertices, 
                                          src[hemind[label.hemi]]['vertno'])
                    
                flips.append(pd.Series(flip, index=[label.name + '_%06d' % vert 
                                                    for vert in vertices]))
            
    return pd.concat(flips)


def find_mni_coordinates(label, parc='HCPMMP1_5_8'):
    """Returns MNI-coordinates of center of mass for selected labels."""
    
    if type(label) is mne.label.Label:
        labels = [label]
    else:
        labels = mne.read_labels_from_annot(
                'fsaverage', parc=parc, regexp=label)
    
    mnicoords = {}
    for label in labels:
        if label.name.startswith('L'):
            hemi = 0
        else:
            hemi = 1
            
        vertno = label.center_of_mass()
        
        mnicoords[label.name] = mne.vertex_to_mni(vertno, hemi, 'fsaverage')
        
    return mnicoords


if 'Brain' in locals():
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
    
    
    def show_label_vertices(src_df, brain, measure=None, parc='HCPMMP1', 
                            fmin=None, fmid=None, fmax=None, center=None, 
                            transparent=True, colormap='auto', 
                            smoothing_steps=5, initial_time=None, 
                            colorbar=True, addlabels=True, label_alpha=0.5):
        """Shows single vertex effects of (a subset of) areas.
        
        src_df has either index with levels [vertex label, time], or an index
        only holding vertex labels; columns contain different measures of the 
        potential effects, or only a single column exists
        """
        if measure is None:
            if type(src_df) is pd.Series:
                src_df = src_df.copy()
            else:
                if src_df.shape[1] > 1:
                    raise ValueError("If no measure is given for selection, "
                                     "the src_df must have only one column")
                else:
                    src_df = src_df.iloc[:, 0].copy()
        else:
            src_df = src_df[measure].copy()
        
        if src_df.index.nlevels == 1:
            # add the time level
            if initial_time is None:
                initial_time = 0
            
            src_df.index = pd.MultiIndex.from_arrays(
                    [src_df.index, np.full_like(src_df.index, initial_time)],
                    names=['label', 'time'])
        
        # only add labels, when this is desired by the user and when there are
        # no labels yet
        addlabels = (len(brain._label_dicts) == 0) and addlabels
        
        if addlabels:
            lnames = src_df.index.levels[0].map(lambda l: l[:-7]).unique()
        
        for hemi in brain.geo.keys():
            data = extract_hemi_data(src_df, hemi)
            
            if data.columns.max() - data.columns.min() > 100:
                time_label_fun = lambda x: '%4d ms' % x
            else:
                time_label_fun = lambda x: '%4d ms' % (x * 1000)
            
            brain.add_data(
                    data, vertices=data.index, hemi=hemi, 
                    smoothing_steps=smoothing_steps, initial_time=initial_time,
                    time=data.columns, time_label=time_label_fun,
                    min=fmin, mid=fmid, max=fmax, center=center, 
                    transparent=transparent, colormap=colormap, 
                    colorbar=colorbar)
            
            if addlabels:
                labels = mne.read_labels_from_annot('fsaverage', parc=parc, 
                                                    hemi=hemi)
                
                hemilnames = lnames[list(lnames.map(
                        lambda s: s[0] == hemi[0].upper()).values)]
                for label in labels:
                    if label.name in hemilnames:
                        brain.add_label(label, borders=True, hemi=hemi, 
                                        alpha=label_alpha)
        
    
    def show_labels_as_data(src_df, measure, brain, labels=None, transform=None,
                            parc='HCPMMP1', fmin=None, fmid=None, fmax=None, 
                            center=None, transparent=True, colormap='auto',
                            initial_time=None, colorbar=True,
                            threshold=0, region_aggfun=None, time_label='auto'):
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
        values = src_df[measure].copy()
        
        # is src_df masked using nans to indicate insignificant data?
        nan_masked = values.isnull().any()
        
        # remove 'insignificant' data, if not done already with a nan mask
        if threshold > 0 and not nan_masked:
            values = ss.null_inconsistent_measure(values, threshold)
        
        # replace nans with base values for that measure
        if nan_masked:
            values = values.fillna(measure_basevals[measure])
            
        # aggregate areas into regions
        if region_aggfun is not None:
            values = set_regions(values, region_aggfun)
        
        if labels is None:
            # assume that no labels shown in brain, yet
    #        brain.add_annotation(parc)
            
            labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both')
            labels = {l.name: l for l in labels}
            
        if len(labels) != src_df.index.get_level_values('label').unique().size:
            raise ValueError('Number of labels ({}) does not fit to number of '
                             'sources in data ({})!'.format(len(labels),
                             src_df.index.levels[0].size))
        
        if time_label == 'auto':
            if (  values.index.levels[1].max() 
                - values.index.levels[1].min()) > 100:
                time_label = lambda x: '%4d ms' % x
            else:
                time_label = lambda x: '%4d ms' % (x * 1000)
    
        for hemi in brain.geo.keys():
            # create data from source estimates
            V = brain.geo[hemi].x.size
            data = np.zeros((V, T))
            for name in [l for l in labels.keys() if l.endswith(hemi)]:
                data[labels[name].vertices, :] = values.xs(name, level='label')
            
            # plot
            brain.add_data(data, fmin, fmax, mid=fmid, center=center,
                           transparent=transparent, colormap=colormap, 
                           time=src_df.index.levels[1].values, 
                           time_label=time_label,
                           colorbar=colorbar, hemi=hemi, 
                           remove_existing=True)
            
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
        
        print(data)
        
        for index, value in data.iteritems():
            if index[0].startswith('L'):
                hemi = 'left'
            else:
                hemi = 'right'
            
            print(index[0])
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
    
    if sys.version_info < (3,):
        brain = stc.plot(subject='fsaverage', surface='inflated', hemi='both',
                         colormap=masked_hot, transparent=False, 
                         clim={'kind': 'value', 'lims': [-1, 0, 1]},
                         initial_time=0.35, smoothing_steps=5)
