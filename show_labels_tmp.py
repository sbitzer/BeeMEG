import source_visualisations as sv
import source_statistics as ss

from surfer import Brain
from surfer import TimeViewer


#%% set basefile and get threshold
measure = 'mu_p_large'

# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201707031206.h5'

# label mode = mean, baseline (-0.3, 0), only first 3 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, motoprep, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201709011758.h5'

# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
basefile = 'source_sequential_201709061827.h5'

#with pd.HDFStore('data/inf_results/' + basefile, 'r') as store:
#    regressors = store.first_level.columns.levels[2]
regressors = ['accev', 'dot_x', 'dot_y', 'abs_dot_x', 'abs_dot_y', 
              'sum_dot_y_prev']
#regressors = ['response', 'motoprep', 'entropy', 'trial_time']

alpha = 0.05

threshold, measure_cdf = ss.find_slabs_threshold(
    basefile, measure, quantile=1-alpha, regressors=regressors, 
    verbose=1, return_cdf=True)

#threshold = 0.5


#%% identify significant clusters
clusters = ss.get_fdrcorr_clusters(basefile, regressors, measure, threshold, 
                                   measure_cdf, fdr_alpha=0.001)


#%% load specific regressor
r_name = 'dot_x'
show_measure = 'mu_mean'

src_df_masked = ss.load_src_df(basefile, r_name, clusters)

brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', foreground='k')

labels = sv.show_labels_as_data(src_df_masked, show_measure, brain, 
                                transparent=True)

#brain.scale_data_colormap(src_df_masked[show_measure].min(),
#                          src_df_masked[show_measure].median(), 
#                          src_df_masked[show_measure].max(), True)

brain.scale_data_colormap(0.01, 0.025, 0.06, True)

#labels = sv.show_labels_as_data(src_df_masked, 'mu_mean', brain, 
#                                transparent=True, 
#                                region_aggfun=lambda a: np.max(a, axis=0))

#tv = TimeViewer(brain)

