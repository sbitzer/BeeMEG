import source_visualisations as sv
import pandas as pd
from surfer import Brain
from surfer import TimeViewer
import numpy as np
import matplotlib.pyplot as plt

r_name = 'dot_x'
measure = 'mu_p_large'
basefile = 'source_HCPMMP1_allsubs_201703301614.h5'
srcfile = basefile[:-3] + '_slabs_%s.h5' % r_name
file = 'mne_subjects/fsaverage/bem/' + srcfile
src_df = pd.read_hdf(file, 'second_level_src')

#brain = Brain('fsaverage', 'both', 'inflated', cortex='low_contrast',
#              subjects_dir=sv.subjects_dir)

threshold = sv.find_slabs_threshold(basefile, measure, quantile=0.95)



#labels = sv.show_labels_as_data(src_df, measure, brain, transparent=True,
#                                threshold=threshold, 
#                                region_aggfun=lambda a: np.max(a, axis=0))
#
#tv = TimeViewer(brain)


#%% show region time courses
values = src_df[measure]

# remove 'insignificant' data
values = sv.null_inconsistent_measure(values, threshold)

region_aggfun=lambda a: np.max(a, axis=0)

regions = pd.DataFrame(
        [], dtype=float, index=values.index.levels[1], columns=
        pd.MultiIndex.from_product([['L', 'R'], sv.Glasser_sections.name], 
                                   names = ['hemi', 'region']))
for section in sv.Glasser_sections.index:
    # find all areas in region
    areas = sv.Glasser_areas[sv.Glasser_areas['main section'] == section]
    
    for hemi in ['l', 'r']:
        # create full label names
        labels = areas['area name'].map(
                lambda s: '%s_%s_ROI-%sh' % (hemi.upper(), s, hemi))
        
        # set values of all areas to that of aggregated value
        avals = values.loc[(labels, slice(None))]
        avals = avals.reset_index(level='time').pivot(columns='time')
        regions.loc[:, (hemi.upper(), sv.Glasser_sections.loc[section, 'name'])] = (
                region_aggfun(avals).values)

active_regions = regions.drop(regions.columns[regions.sum() == 0], axis=1)

for hemi in ['L', 'R']:
    active = active_regions.loc[:, hemi]
    N = active.shape[1]
    fig, axes = plt.subplots(int(np.ceil(N/3.0)), 1, sharex=True)
    
    for i, ax in enumerate(axes):
        active.iloc[:, slice(i*3, (i+1)*3)].plot.line(ax=ax)
        ax.set_ylim([threshold, 1.0])
        
    axes[0].set_title('hemisphere: ' + hemi)
    
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

