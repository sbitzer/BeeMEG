#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:42:26 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import mne
import helpers
import os
import gc
import source_statistics as ss
import regressors
import seaborn as sns
import matplotlib.pyplot as plt

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% options
# label mode= None, lselection=pre-motor and motor areas, window=[0.3, 0.9]
srcfile = 'source_epochs_allsubs_HCPMMP1_201710231731.h5'

# times after dot onset (in ms) to investigate
loadtimes = [320, 400]

# regressor name to investigate
# the first is the main one to investigate, following ones are used for comparison
r_names = ['saccade_x', 'dot_x', 'dot_y', 'sum_dot_x']

# dots to investigate
dots = np.arange(1, 6)


#%% find eye and arm vertices of primary motor cortex
eyelabels = {}
armlabels = {}
for hemi in ['lh', 'rh']:
    mlabels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_motor', 
                                         hemi=hemi)
    m1label, = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_5_8', 
                                         hemi=hemi, regexp='_4_')
    
    get_label = lambda pat: [label for label in mlabels 
                             if label.name.find(pat) >= 0][0]
    
    eyelabels[hemi] = mne.Label(
            np.intersect1d(m1label.vertices, get_label('Ocular').vertices),
            hemi=hemi, name='M1_eye', subject=m1label.subject)
    armlabels[hemi] = mne.Label(
            np.intersect1d(m1label.vertices, 
                           get_label('UpperExtremity').vertices),
            hemi=hemi, name='M1_arm', subject=m1label.subject)


#%% load data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

srcfile = os.path.join(ss.bem_dir, srcfile)

epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
epochs_std = pd.read_hdf(srcfile, 'epochs_std')

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select('label_tc', 'subject=2')

# find the eye and arm vertices in the columns
# note that there may be vertices belonging not to M1 (area 4) according to the
# column labels, this can happen when you use HCPMMP1 in the srcfile and 
# HCPMMP1_5_8 here, differences between them are small, but individual vertices
# at the borders of areas may be assigned to their neighbours, e.g., the M1
# arm area defined by HCPMMP1_5_8 contains one source vertex that apparently 
# belongs to area 3a in HCPMMP1
armcols = {}
eyecols = {}
for hemi in ['lh', 'rh']:
    columns = epochs.columns[epochs.columns.map(
            lambda c: c.startswith(hemi[0].upper())).values.astype(bool)]
    vertices = columns.map(lambda s: int(s[-6:]))
        
    vert = armlabels[hemi].get_vertices_used(vertices)
    armcols[hemi] = columns[vertices.get_indexer_for(vert)]
    
    vert = eyelabels[hemi].get_vertices_used(vertices)
    eyecols[hemi] = columns[vertices.get_indexer_for(vert)]

loadlabels = np.r_[armcols['lh'].values, armcols['rh'].values,
                   eyecols['lh'].values, eyecols['rh'].values]

times = epochs.index.levels[2]

def extract_time_data(epochs, time):
    # get the times associated with the dots
    datat = time + (dots - 1) * helpers.dotdt * 1000
    assert np.all(datat <= times[-1]), ("The times where you are looking for an "
                  "effect have to be within the specified time window")
    datat = times[[np.abs(times - t).argmin() for t in datat]]

    data = (  epochs.loc[(slice(None), datat), :] 
            - epochs_mean[loadlabels]) / epochs_std[loadlabels]
    
    data.index = pd.MultiIndex.from_product([epochs.index.levels[0], dots], 
                                            names=['trial', 'dot'])
    
    return data

def load_normalised_data(sub):
    with pd.HDFStore(srcfile, 'r') as store:
        epochs = store.select('label_tc', 'subject=sub')
    
    return pd.concat([extract_time_data(epochs.loc[sub, loadlabels], time) 
                      for time in loadtimes],
                      keys=loadtimes, names=['time', 'trial', 'dot'])

data = pd.concat([load_normalised_data(sub) for sub in subjects],
                 keys=subjects, names=['subject', 'time', 'trial', 'dot'])

data = data.reorder_levels(['time', 'subject', 'trial', 'dot'])
data.sort_index(inplace=True)

gc.collect()

print('loaded data for M1 eye and arm regions')


#%% load regressors

# get first regressor in new dataframe with appropriate index
reg = pd.concat(
        [regressors.trial_dot[r_names[0]].loc[(slice(None), list(dots))]]
        * subjects.size, keys=subjects, names=['subject', 'trial', 'dot'])
reg = pd.DataFrame(reg)

# fill in the remaining regressors only copying values
for name in r_names[1:]:
    reg[name] = np.tile(
            regressors.trial_dot[name].loc[(slice(None), list(dots))].values, 
            subjects.size)
   
    
#%% find the peak vertices in eye and arm regions
#   simply based on correlation across all time points

# use dot_x for this as conservative choice: if I then find benefits of 
# sum_dot_x in the selected signal, this is stronger evidence for sum_dot_x
r_name = 'dot_x'

corrs = data.apply(
        lambda dat: np.corrcoef(
                dat, np.tile(reg[r_name].values, len(loadtimes)), 
                rowvar=False)[0, 1])

peaks = {'L': {}, 'R': {}}
for hemi in ['lh', 'rh']:
    peaks[hemi[0].upper()]['arm'] = corrs[armcols[hemi]].abs().argmax()
    peaks[hemi[0].upper()]['eye'] = corrs[eyecols[hemi]].abs().argmax()
    

#%% plot relation between signal and regressor
r_name = 'dot_x'

time = loadtimes[-1]

colors = sns.cubehelix_palette(2, start=3, rot=0.5, light=.75, dark=.25)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7.5, 4.5])

xrange = reg[r_name].quantile([0.01, 0.99])
xpred = np.linspace(*xrange, 200)[:, None]

for hemi, ax in zip(['L', 'R'], axes):
    print('processing hemisphere %s ...' % hemi, flush=True)
    
    for part, color in zip(['arm', 'eye'], colors):
        gpm = ss.fitgp(reg[r_name], data.loc[time, peaks[hemi][part]], 
                       np.linspace(*xrange, 15))
        
        ss.plot_gp_result(ax, xpred, gpm, color, part)
    
    ax.set_title(hemi)
    ax.set_xlabel('x-coordinate (px)')
        
axes[0].legend(loc='upper left')
axes[0].set_ylabel('normalised mean currents in peak vertex')

fig.subplots_adjust(top=.92, right=.96, wspace=.12)
fname = os.path.join(figdir, 'correlation_arm_vs_eye_%s_%d.png' % (
        r_name, time))
fig.savefig(fname, dpi=300)