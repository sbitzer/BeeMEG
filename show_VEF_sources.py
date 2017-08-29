#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
to interact with Mayavi's mlab call ipython using (requires wxpython):

    ETS_TOOLKIT=wx ipython --gui=wx

probably QT is better, but I couldn't get it to work properly
see http://docs.enthought.com/mayavi/mayavi/mlab.html#simple-scripting-with-mlab
for further information about the pitfalls in trying to run mlab interactively

Created on Fri Aug 18 14:40:28 2017

@author: bitzer
"""

from __future__ import print_function
import mne
import numpy as np
import pandas as pd
import matplotlib as mpl
from mayavi import mlab

from surfer import Brain


def extract_hemi_data(src_df, measure, time, hemi):
    data = pd.DataFrame(src_df[measure].copy())
    data['hemi'] = data.index.map(lambda x: x[0].lower()+'h')
    data.set_index('hemi', append=True, inplace=True)
    data.reset_index(0, inplace=True)
    data['level_0'] = data['level_0'].map(lambda x: int(x[-6:]))
    data.set_index('level_0', append=True, inplace=True)
    data.sort_index(inplace=True)
    
    return data.loc[hemi].values, data.loc[hemi].index.values


def morph_colortable(table, clim):
    table_new = table.copy()
    
    n_colors = table.shape[0]
    n_colors2 = int(n_colors / 2)
    
    # control points in data space
    fmin, fmid, fmax = clim
    
    # Index of fmid in new colorbar (which position along the N colors would
    # fmid take, if fmin is first and fmax is last?)
    fmid_idx = int(np.round(n_colors * ((fmid - fmin) /
                                        (fmax - fmin))) - 1)
    
    # morph each color channel so that fmid gets assigned the middle color of
    # the original table and the number of colors to the left and right are 
    # stretched or squeezed such that they correspond to the distance of fmid 
    # to fmin and fmax, respectively
    for i in range(4):
        part1 = np.interp(np.linspace(0, n_colors2 - 1, fmid_idx + 1),
                          np.arange(n_colors),
                          table[:, i])
        table_new[:fmid_idx + 1, i] = part1
        part2 = np.interp(np.linspace(n_colors2, n_colors - 1,
                                      n_colors - fmid_idx - 1),
                          np.arange(n_colors),
                          table[:, i])
        table_new[fmid_idx + 1:, i] = part2
        
    return table_new


def morph_divergent_cmap(cmap, clim):
    if len(clim) == 3:
        if clim[0] == 0:
            clim = np.r_[-np.flipud(clim[1:]), clim]
        else:
            raise ValueError("For divergent colormap clim needs to be either "
                             "of length 5, or the inflection point at clim[0] "
                             " must be 0.")
    else:
        raise ValueError("For divergent colormap clim needs to be either "
                             "of length 5, or the inflection point at clim[0] "
                             " must be 0.")
    
    n_colors = cmap.shape[0]
    
    return np.r_[morph_colortable(cmap[:int(n_colors/2), :], clim[:3]), 
                 morph_colortable(cmap[int(n_colors/2):, :], clim[2:])]


#%% load results from notebook and plot them
measure = 'tval'
hemi = 'rh'
contrast = 'right_close'

try:
    src_df = pd.read_hdf('VEFdata_tmp.h5', contrast)
except KeyError:
    src_df = pd.read_hdf('VEFdata_tmp.h5', 'second_level')
    src_df = src_df.xs(contrast, axis=1, level='regressor')

if 'mlog10p' not in src_df.columns:
    src_df['mlog10p'] = -np.log10(src_df.pval)

time = pd.read_hdf('VEFdata_tmp.h5', 'options').time / 1000.

# make my own transparent colormap, because MNE's is crap
cmap = mpl.cm.coolwarm_r(np.arange(256))
cmap[:, -1] = np.r_[np.ones(64), np.linspace(1, 0, 64),
                    np.linspace(0, 1, 64), np.ones(64)]
cmap = cmap * 255

views = {'rh': (-128, 99, 431, np.array([ 0.,  0.,  0.])),
         'lh': (-54, 99, 431, np.array([ 0.,  0.,  0.]))}

def plot_V1V2(measure, hemi, clim):
    data, vert = extract_hemi_data(src_df, measure, time, hemi)
    
    brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                  subjects_dir='mne_subjects', background='w', foreground='k')
    
    brain.add_data(data, vertices=vert, min=-clim[2], max=clim[2],
                   time=[time], time_label=lambda t: '%d ms' % (t*1000),
                   colormap=morph_divergent_cmap(cmap, clim), 
                   hemi=hemi, smoothing_steps=5)
    
    labels = mne.read_labels_from_annot(
        'fsaverage', parc='HCPMMP1', hemi=hemi, regexp='[LR]_((V1)|(V2)).*')
    
    for label in labels:
        brain.add_label(label, borders=True)
        
    mlab.view(*views[hemi])
    
    return brain


# choose control points
clim = [0, 2., 5]

brain = plot_V1V2(measure, hemi, clim)

brain.save_single_image('/home/bitzer/ZIH/texts/BeeMEG/figures/V1V2_%s_%d_%s.png' % (contrast, time*1000, hemi))