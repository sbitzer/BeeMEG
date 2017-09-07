#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:05:45 2017

@author: bitzer
"""

import pandas as pd
import source_visualisations as sv

from surfer import Brain


#%% load results

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709041745.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709051204.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount, accev_time delay 0
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709071202.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount, accev_time delay 0.1
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709061841.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount, accev_time delay 0.2
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_time_201709071302.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount, accev_time delay 0.3
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709061815.h5'

# label mode = mean, baseline (-0.3, 0)
# source GLM, motoprep, motoresponse, dotcount, accev_time delay 0.4
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_time_201709071413.h5'

second_level = pd.read_hdf('data/inf_results/' + basefile, 'second_level')


#%% show results
brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', foreground='k')

def plot(r_name, measure, mask=None, clim=[0, 3., 5], perm=0):
    src_df = second_level.loc[perm].xs(r_name, axis=1, level='regressor')
    
    if mask:
        insig = pd.np.logical_not(second_level.loc[perm, ('significant', r_name)])
        src_df[insig.values] = 0
    
    src_df['time'] = 0
    src_df.set_index('time', append=True, inplace=True)
    
    if measure in ['tval', 'mean']:
        labels = sv.show_labels_as_data(src_df, measure, brain, 
                                        colormap='divtrans', clim=clim)
    else:
        labels = sv.show_labels_as_data(src_df, measure, brain)
    
    return labels

r_name = 'accev_time'
show_measure = 'tval'

labels = plot(r_name, show_measure, True)


#%% list significant areas together with their regions
def sigareas(r_name, perm=0):
    src_df = second_level.loc[perm].xs(r_name, axis=1, level='regressor')

    src_df = src_df[src_df.significant.values]
    
    src_df['region'] = list(map(sv.ss.get_Glasser_section, src_df.index))
    
    return src_df
