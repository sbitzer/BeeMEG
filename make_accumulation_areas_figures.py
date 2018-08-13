#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:29:24 2018

@author: bitzer
"""

import pandas as pd
import numpy as np
import os, sys
from sklearn.cluster import KMeans

import source_statistics as ss

if sys.version_info < (3,):
    from surfer import Brain
    import source_visualisations as sv

    figdir = sv.fig_dir


#%% load source-level results
# baseline (-0.3, 0), only first 3 dots, trialregs_dot=3, move_dist, 
# sum_dot_y, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201707031206.h5'

# label mode = mean, baseline (-0.3, 0), first 5 dots, 
# trialregs_dot=0, source GLM, sum_dot_y, constregs=0 for 1st dot, 
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201709061827.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials, times and subjects
#basefile = 'source_sequential_201801261754.h5'

# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
#basefile = 'source_sequential_201801291241.h5'

# loose source orientations, but only their overall magnitude
# label mode = mean, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
#basefile = 'source_sequential_201807031144.h5'

# loose source orientations, but cortex normal currents
# label mode = mean_flip, baseline (-0.3, 0), all dots with toolate=-200, 
# time window [0, 690], exclude time-outs, local normalisation of DM
# trialregs_dot=0, accev, sum_dot_y_prev, percupt, constregs=0 for 1st dot, 
# label_tc normalised across trials but within times and subjects
basefile = 'source_sequential_201807041930.h5'

# newest results are based on this parcellation, cf. 'srcfile' in basefile
parc = 'HCPMMP1_5_8'

regressors = ['dot_x', 'accev']

measure = 'tval'

twin = dict(peak1=[slice(110, 130)], 
            peak2=[slice(160, 200)],
            between=[slice(100, 100), slice(140, 150), slice(210, 290)],
            plateau=[slice(300, 500)])
wnames = pd.Index(twin.keys()).sort_values()

accev_off = 100
accev_sign = -1

fdr_alpha = 0.05

second_level_perms = pd.read_hdf(os.path.join(ss.inf_dir, basefile), 
                           'second_level')[measure][regressors]
second_level = second_level_perms.loc[0]

P = second_level_perms.index.get_level_values('permnr').unique().size

times = second_level.index.get_level_values('time').unique()
labels = second_level.index.get_level_values('label').unique()


#%% find brain areas with significant effects in selected time windows
clusters = ss.get_fdrcorr_clusters(basefile, regressors, fdr_alpha, 
                                   use_basefile=True)

srcdf_masked = pd.concat(
        [ss.load_src_df(basefile, reg, clusters, use_basefile=True) 
         for reg in regressors],
        keys=regressors, names=['regressor', 'label', 'time'])
srcdf_masked.dropna(inplace=True)

is_significant = pd.DataFrame(
        np.zeros((len(regressors) * len(labels), len(wnames)), dtype=bool),
        index=pd.MultiIndex.from_product(
                [regressors, labels], names=['regressor', 'label']),
        columns=wnames)
    
for name in wnames:
    for win in twin[name]:
        for reg in regressors:
            if reg == 'accev':
                win2 = slice(win.start - accev_off, win.stop - accev_off)
            else:
                win2 = win
                
            sigl = (srcdf_masked.loc[(reg, slice(None), win2)]
                    .index.get_level_values('label').unique())
            is_significant.loc[(reg, list(sigl)), name] = True


#%% compute difference in measure magnitudes between dot_x and accev
x_m_a = (second_level.loc[(slice(None), slice(accev_off, None)), 
                          'dot_x'].abs()
         - second_level.loc[(slice(None), slice(0, times[-1] - accev_off)), 
                            'accev'].abs().values)

difftimes = x_m_a.index.get_level_values('time').unique()


#%% aggregate differences within defined time windows
xma_win = pd.DataFrame(
        np.zeros((labels.size, wnames.size)), dtype=float, 
        index=labels, columns=wnames)

for name in wnames:
    for win in twin[name]:
        xma_win[name] += x_m_a.loc[(slice(None), win)].groupby('label').sum()


#%% estimate time points at which area appears to represent accumulated 
#   evidence for the first time after dot onset
accev_labels = xma_win['between'][xma_win['between'] < -10].index
first_accev = pd.Series(np.zeros(accev_labels.size), dtype=float, 
                        index=accev_labels)
for label in accev_labels:
    diffs = (second_level.loc[(label, slice(accev_off, None)), 'dot_x']
             - accev_sign * second_level.loc[
                     (label, slice(0, times[-1] - accev_off)), 'accev']
             .values).abs()[:, None]
    
    fit = KMeans(n_clusters=2).fit(diffs)
    
    cids = pd.Series(fit.predict(diffs), index=difftimes)
    
#    fig, ax = plt.subplots(); 
#    ax.plot(cids + np.random.randn(diffs.size) * 0.01, diffs, '.'); 
#    ax.set_xlim(-1, 2)
    
    accid = cids.loc[slice(100, 150)].mean() > 0.5
    
    cids = cids.loc[slice(210, None)]
    
    first_accev.loc[label] = cids[cids != accid].index[0]
        
        
#%% visualising results in brain
if sys.version_info < (3,):
    def get_colorinfo(srcdf, quantiles=[0.8, 0.95, 1]):
        qvals = srcdf.abs().quantile(quantiles)
        
        return {'fmin': qvals.loc[quantiles[0]].values[0],
                'fmid': qvals.loc[quantiles[1]].values[0],
                'fmax': qvals.loc[quantiles[2]].values[0],
                'transparent': True,
                'center': 0,
                'colormap': 'auto'}
    
    filepat_base = 'xma_brain_' + measure
    
    def brain_plot(wname, brain, toplabels=None, save=False, ):
        views = {'rh': ['medial', 'lateral'], #{'azimuth': -20, 'elevation': 62, 'roll': -68}], 
                 'lh': ['lateral', 'medial']}
        
        if type(brain) is str:
            hemi = brain
            brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
                          subjects_dir=sv.subjects_dir, background='w', 
                          foreground='k')
        else:
            hemi = brain.geo.keys()[0]
            
        srcdf = pd.DataFrame(xma_win[wname].values.copy(), 
                             index=pd.MultiIndex.from_product(
                                     [labels, [0]], names=['label', 'time']),
                             columns=[measure])
        
        # get colorinfo based on distribution of differences before nulling
        cinfo = get_colorinfo(srcdf)
                             
        # set all areas to 0 for which there is insufficient evidence that they
        # represent evidence; for positive differences supporting dot_x only 
        # consider significant dot_x areas and equally for accev and negative
        for reg in regressors:
            if reg == 'accev':
                siglabels = srcdf[srcdf[measure] < 0].xs(0, level='time').index
            else:
                siglabels = srcdf[srcdf[measure] > 0].xs(0, level='time').index
            siglabels = is_significant.xs(reg, level='regressor').loc[
                    list(siglabels), wname]
            siglabels = siglabels[~siglabels].index
            srcdf.loc[(list(siglabels), 0), measure] = 0
            
        sv.show_labels_as_data(srcdf, measure, brain, time_label=None, 
                               parc=parc, **cinfo)
        
#        if toplabels is not None:
#            alllabels = mne.read_labels_from_annot(
#                    'fsaverage', parc=parc, hemi=hemi)
#            for label in alllabels:
#                if label.name in toplabels[hemi]:
#                    brain.add_label(label, borders=1, hemi=hemi, alpha=0.8, 
#                                    color='k')
        
        # increase font size of colorbar - this only works by increasing the 
        # colorbar itself and setting the ratio of colorbar to text
        brain.data['colorbar'].scalar_bar.bar_ratio = 0.35
        brain.data['colorbar'].scalar_bar_representation.position = [0.075, 0.01]
        brain.data['colorbar'].scalar_bar_representation.position2 = [0.85, 0.12]
        
        if save:
            filepat = os.path.join(figdir, filepat_base + '_{}_{}.png'.format(
                    hemi, wname))
            
            brain.save_montage(filepat, views[hemi], colorbar=0)
        
        return brain, srcdf[srcdf[measure].abs() > cinfo['fmid']].xs(
                0, level='time')
    
    
    #%%
    hemi = 'rh'
    brain, areas = brain_plot('peak1', hemi)