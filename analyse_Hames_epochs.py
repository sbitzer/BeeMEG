#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:05:37 2017

@author: bitzer
"""

import numpy as np
import mne
from scipy.io import loadmat
import helpers
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import collections


#%%
subjects = None

if subjects is None:
    subjects = helpers.find_available_subjects('data/meg_behav', 
                                               'data/meg_final_data')
subjects = np.sort(np.array(subjects))
S = subjects.size
if S > 1:
    fname = pd.datetime.now().strftime('clusres_5thdot_meg_%Y%m%d%H%M')

# create events array
events = np.c_[np.arange(480), np.full(480, 301, dtype=int), 
               np.zeros(480, dtype=int)]

# type of event
event_id = {'dot_onset': 0}

# when does an epoch start? - 0.3 s before onset of the first dot
tmin = -0.3

# channel connectivity from FieldTrip for cluster analysis
connectivity, co_channels = mne.channels.read_ch_connectivity(
    '/home/bitzer/proni/BeeMEG/MEG/neuromag306mag_neighb.mat')


#%% build parametric regressor quantifying how much the 5th dot supports the 
#  corrrect decision
dotpos = helpers.load_dots(dotdir='data')
trial_info = helpers.get_5th_dot_infos(dotpos)

names = ['intercept', 'support_correct']
#design_matrix = np.c_[np.ones(480), trial_info['support_correct']]
design_matrix = np.c_[np.ones(480), np.sign(trial_info['support_correct'])]
#design_matrix[:, 1] = design_matrix[np.random.randint(480, size=480), 1]


#%% run subject specific analysis
max_cluster_p = pd.Series([], name='p of largest cluster', 
                          index=pd.Index([], name='subject'))
clus_res = {}
cluster_result = collections.namedtuple('cluster_result', 
                                        ['T_obs', 'clusters', 'pval', 'H0'])
for sub in subjects:
    print('\nSubject %02d' % sub)
    print('----------')
    
    try:
        # raw file of subject, block 1
        rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm%02da/bm%02da1.fif' % (sub, sub)
        
        # get MEG-info from raw file (for channel layout and names)
        info = mne.io.read_info(rawfile)
    except FileNotFoundError:
        # just use last working info - for now it shouldn't matter
        pass

    # Hame has downsampled to 4ms per sample
    info['sfreq'] = 1 / 0.004

    # load Hame's epoched data
    # conveniently, the order of the channels in the matlab data is the same as
    # the order given by the raw fif-file (see channel_names.mat which I extracted
    # from variable hdr in one of the files in meg_original_data)
    mat = loadmat('data/meg_final_data/%02d_sdat.mat' % sub)

    # construct full data array corresponding to the channels defined in info
    data = np.zeros((480, 313, 301))
    # MEG data
    data[:, :306, :] = mat['smeg']
    # EOG + ECG
    data[:, 306:309, :] = mat['setc'][:, :3, :]
    # triggers (STI101) - Hame must have added this manually as I can't see where
    # it was added in her Matlab scripts, but the time-courses and values 
    # definitely suggest that it's STI101
    # NOT PRESENT FOR ALL SUBJECTS
    if mat['setc'].shape[1] == 4:
        data[:, 310, :] = mat['setc'][:, 3, :]

    # create MNE epochs
    epochs = mne.EpochsArray(data, info, events, tmin, event_id, proj=False)
    
    # do extra analyses for single subject
    if S == 1:
        lm = mne.stats.regression.linear_regression(epochs, design_matrix, names)

        #%% plot topomap of result
        support_correct = lm['support_correct']
        
        ch_type = 'mag'
        times = np.linspace(0, .7, 15)
        times = 'peaks'
        
        # plot p-values
        fig = support_correct.mlog10_p_val.plot_topomap(ch_type=ch_type, size=1.5, 
            unit='-log10_p', vmin=0, vmax=4.0, scale=1, times=times);
        
        # extract times from titles
        times = [float(ax.get_title()[:-3]) / 1000 for ax in fig.axes[:-1]]
        
        # plot t-values
        support_correct.t_val.plot_topomap(ch_type=ch_type, size=1.5, unit='t', 
            vmin=-4.0, vmax=4.0, scale=1, times=times);
        
                                           
        #%% check linear regression manually
        evo = support_correct.mlog10_p_val
        chname, timeind = evo.get_peak('mag', time_as_index=True)
        chind = np.flatnonzero(np.array(evo.ch_names) == chname)[0]
        
        df = pd.DataFrame(np.c_[data[:, chind, timeind], design_matrix[:, 1]], 
                          columns=['data', 'support_correct'])
        
        res = smf.ols('data ~ support_correct', data=df).fit()
        
        print(res.summary())
        
        beta, stderr, t_val, p_val, mlog10_p_val = mne.stats.regression._fit_lm(
            data, design_matrix, names)
        
        print('\n')
        print('t-val statsmodels: {}\n'
              't-val linregress:  {}\n'
              't-val mne:         {}\n'.format(res.tvalues['support_correct'],
              helpers.linregress_t(data[:, chind, timeind][:, None], design_matrix[:, 1])[0], 
              support_correct.t_val.data[chind, timeind]))
        
        
        #%% permutation test of identified peak
        rind = np.random.randint(480, size=(480, 10000))
        perm_t_vals = helpers.linregress_t(data[:, chind, timeind][rind], 
                                           design_matrix[:, 1])
        
        ax = sns.distplot(perm_t_vals)
        ax.plot(support_correct.t_val.data[chind, timeind], 0, 'r*')
        
        perm_p_val = np.mean(np.abs(perm_t_vals) >= 
                             np.abs(support_correct.t_val.data[chind, timeind]))
        print('permuted p-value:   {}\n'
              'regression p-value: {}'.format(perm_p_val, 
              support_correct.p_val.data[chind, timeind]))


    #%% permutation cluster analysis
    epochs_mag = epochs.pick_types(meg='mag')
    
    # channels must have the same order
    assert(np.all(np.array(epochs_mag.info['ch_names']) == co_channels))
    
    T_obs, clusters, pval, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        epochs_mag.get_data().transpose(0, 2, 1), connectivity=connectivity, 
        stat_fun=lambda data: helpers.linregress_t(data, design_matrix[:, 1]), 
        tail=0, n_permutations=1024, threshold=3, max_step=1)
    
    max_cluster_p.loc[sub] = pval.min()
    clus_res[sub] = cluster_result(T_obs, clusters, pval, H0)
    
    if S == 1:
        ax = sns.distplot(H0, kde=False)
        ax.plot([np.sum(T_obs[clus]) for clus in clusters], np.zeros(len(clusters)), '*r')
    else:
        np.savez_compressed('data/inf_results/' + fname, 
                            max_cluster_p=max_cluster_p, clus_res=clus_res)


#%% show distribution of p-values (should be uniform in [0,1] under H0)
if S > 1:
    sns.distplot(max_cluster_p, kde=False, rug=True)