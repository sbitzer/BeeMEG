#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:19:02 2018

@author: bitzer
"""

from __future__ import print_function
import helpers
import source_visualisations as sv
import source_statistics as ss
import numpy as np
import pandas as pd
import os
import mne
import scipy.stats

try:
    from surfer import Brain
except:
    pass

figdir = helpers.figdir


#%% some options
# newest results are based on this parcellation, cf. 'srcfile' in basefile
parc = 'HCPMMP1_5_8'

r_name = 'response'

if r_name == 'response':
    # loose source orientations, but cortex normal currents
    # baseline (-0.3, 0), response-aligned
    # time [-1000, 500], exclude time-outs
    # trial_time, intercept, response
    # trial normalisation of data, local normalisation of DM
    basefile = 'source_singledot_201808171756.h5'
    
    twins = [("build-up", [-500, -120]),
             ("response", [-30, 100])]

winnames = [win[0] for win in twins]

show_measure = 'tval'


#%% prepare data
data = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 'first_level')
data = data.loc[0].xs('beta', level='measure', axis=1).xs(
        r_name, level='regressor', axis=1).copy()

# scale data so that for each time point and subject the mean magnitude of
# betas across brain areas is 1
data = data.unstack('time')
datanorm = data / data.abs().mean()
datanorm = datanorm.stack('time')

# average within the time windows
avdata = pd.concat(
        [datanorm.loc[(slice(None), slice(*(win[1]))), :].groupby('label').mean() 
         for win in twins], keys=winnames, 
        names=['window', 'label'])

# calculate magnitude(!) difference
diff = avdata.loc[winnames[0]].abs() - avdata.loc[winnames[1]].abs()

# perform t-test
tvals, pvals = scipy.stats.ttest_1samp(diff, 0, axis=1)

# put results into standard container
srcdf = pd.DataFrame(
        np.c_[diff.mean(axis=1), diff.std(axis=1), tvals, pvals],
        index=pd.MultiIndex.from_product(
                [diff.index, [0]],
                names=['label', 'time']),
        columns=['mean', 'std', 'tval', 'pval'])


#%% show results in brain
if show_measure not in srcdf.columns:
    ss.add_measure(srcdf, show_measure)

hemi = 'lh'

brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', foreground='k')

sv.show_labels_as_data(srcdf, show_measure, brain, time_label=None, parc=parc,
                       center=0)