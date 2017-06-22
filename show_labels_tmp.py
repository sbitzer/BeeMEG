import source_visualisations as sv
import pandas as pd
#from surfer import Brain
#from surfer import TimeViewer
import numpy as np
import matplotlib.pyplot as plt


#%% set basefile and get threshold
measure = 'mu_p_large'

# baseline [-0.3, 0], trialregs_dot=5
# basefile = 'source_HCPMMP1_allsubs_201703301614.h5'

# baseline None, trialregs_dot=5
#basefile = 'source_HCPMMP1_allsubs_201706091054.h5'

# baseline (-0.3, 0), trialregs_dot=5, GLM in source space
#basefile = 'source_sequential_201706141650.h5'

# baseline (-0.3, 0), trialregs_dot=5, GLM in source space, move_dist, sum_dot_y
#basefile = 'source_sequential_201706191442.h5'

# baseline (-0.3, 0), trialregs_dot=5, GLM in source space, move_dist, 
# sum_dot_y, constregs=0 for 1st dot
basefile = 'source_sequential_201706201654.h5'

#brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
#              subjects_dir=sv.subjects_dir)

#with pd.HDFStore('data/inf_results/' + basefile, 'r') as store:
#    regressors = store.first_level.columns.levels[2]
regressors = None
threshold = sv.find_slabs_threshold(basefile, measure, quantile=0.95, 
                                    regressors=regressors, exclude='trialregs',
                                    verbose=1)
#threshold = 0.5


#%% load specific regressor
r_name = 'dot_x'
srcfile = basefile[:-3] + '_slabs_%s.h5' % r_name
file = 'mne_subjects/fsaverage/bem/' + srcfile
src_df = pd.read_hdf(file, 'second_level_src')

#labels = sv.show_labels_as_data(src_df, measure, brain, transparent=True,
#                                threshold=threshold, 
#                                region_aggfun=lambda a: np.max(a, axis=0))
#
#tv = TimeViewer(brain)


#%% show region time courses
#active = sv.get_significant_areas_in_region(src_df[measure], threshold, 3)
active = sv.get_significant_areas_in_time(src_df[measure], threshold, [100, 130])
#active = sv.get_significant_regions(src_df[measure], threshold)

sv.show_timecourses(active, [threshold, 1])
    
plt.show()


#%% 
from itertools import groupby

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

