#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:25:37 2017

@author: bitzer
"""

import source_visualisations as sv
import source_statistics as ss

from surfer import Brain


#%% set basefile and get threshold
measure = 'mlog10p'
perm = 0

# label mode = mean, baseline (-0.3, 0), dots 3-5, 
# source GLM, sum_dot_y_prev_3, accev_3, dot_x, dot_y, trial regressors
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_singledot_201709071126.h5.tmp'

#with pd.HDFStore('data/inf_results/' + basefile, 'r') as store:
#    regressors = store.second_level.columns.levels[1]
regressors = ['accev_3', 'dot_x_3', 'dot_x_4', 'dot_x_5', 'dot_y_3', 'dot_y_4',
              'dot_y_5', 'entropy', 'response', 'sum_dot_y_prev_3',
              'trial_time']

alpha = 0.05

threshold, measure_cdf = ss.find_slabs_threshold(
    basefile, measure, quantile=1-alpha, regressors=regressors, 
    verbose=1, return_cdf=True, use_basefile=True, perm=perm)


#%% identify significant clusters
clusters = ss.get_fdrcorr_clusters(basefile, regressors, measure, threshold, 
                                   measure_cdf, fdr_alpha=0.001, use_basefile=True,
                                   perm=perm)
print('cluster counts:')
print(clusters.label.groupby(level='regressor').count())


#%% load specific regressor
r_name = 'response'
show_measure = 'mlog10p'

src_df_masked = ss.load_src_df(basefile, r_name, clusters, use_basefile=True)

brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', foreground='k')

labels = sv.show_labels_as_data(src_df_masked, show_measure, brain, 
                                transparent=True)

brain.scale_data_colormap(src_df_masked[show_measure].min(),
                          src_df_masked[show_measure].median(), 
                          src_df_masked[show_measure].max(), True)

#brain.scale_data_colormap(0.01, 0.025, 0.06, True)

#tv = TimeViewer(brain)

