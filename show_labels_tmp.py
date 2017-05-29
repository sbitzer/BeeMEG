import source_visualisations as sv
import pandas as pd
from surfer import Brain
from surfer import TimeViewer
import numpy as np

r_name = 'dot_x'
measure = 'mu_p_large'
basefile = 'source_HCPMMP1_allsubs_201703301614.h5'
srcfile = basefile[:-3] + '_slabs_%s.h5' % r_name
file = 'mne_subjects/fsaverage/bem/' + srcfile
src_df = pd.read_hdf(file, 'second_level_src')

brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir)

threshold = sv.find_slabs_threshold(basefile, measure, quantile=0.95)

labels = sv.show_labels_as_data(src_df, measure, brain, transparent=True,
                                threshold=threshold, 
                                region_aggfun=lambda a: np.max(a, axis=0))

tv = TimeViewer(brain)


#%% 
from itertools import groupby
import matplotlib.pyplot as plt

def count_consecutive_nonzeros(iterable):
    return np.array([sum(1 for _ in group) 
                     for label, group in groupby(iterable) 
                     if label!=0])
    
stable_regions = src_df.index.levels[0][
        src_df.mu_p_large.groupby(level='label').agg(
                lambda x: np.any(count_consecutive_nonzeros(x > threshold) > 2)).astype(bool)]
        
def plot_label_tcs(df, labels):
    data = src_df.loc[labels, 'mu_p_large']
    if type(labels) != str:
        data = data.reset_index(level='label').pivot(columns='label')
        
    return plt.plot(df.index.levels[1].values, data)


#%% 
#tclust = src_df.copy()
#tclust.loc[(np.setdiff1d(src_df.index.levels[0], stable_regions), slice(None)),
#           'mu_p_large'] = 0
#la = sv.show_labels_as_data(tclust, 'mu_p_large', brain, transparent=True)

