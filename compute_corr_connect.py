#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:16:56 2018

@author: bitzer
"""

import numpy as np
import helpers
import pandas as pd
import regressors
import os, gc
from datetime import datetime


#%% options and data
nperm = 3

timeslice = [100, 600]

delays = np.arange(-100, 301, 10)

exclude_to = False

toolate = -200

labels = ['6d', 'IP0', 'V3', '7m', '31pd', 'v23ab', 'LO3', 'VMV2', 'FST', 
          'p24pr', '7AL', '3a', 'POS2', '5mv', '4', '6a', 'V1']
labels = np.concatenate(
        [np.array(['%s_%s_ROI-%sh' % (hemi.upper(), l, hemi.lower()) 
                  for l in labels]) 
         for hemi in ['l', 'r']])
labels.sort()

# data to use, 'meg' for MEG channels, 'source' for sources
datatype = 'source'

if datatype == 'meg':
    sfreq = 100
    megtype = False
    eog = True
    bl = (-0.3, 0)
    window = [0, 2.5]
    
    srcfile = helpers.create_meg_epochs_hdf(sfreq, sfreq, window, megtype, bl,
                                            filt_freqs=2, eog=eog)
    epname = 'epochs'
    
elif datatype == 'source':
    # label mode = mean, long epochs, HCPMMP_5_8
#    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201712061743.h5'
    
    # label mode = mean, long epochs, HCPMMP_5_8, loose=0.2, magnitude ori
#    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201807021743.h5'
    
    # label mode = mean_flip, long epochs, HCPMMP_5_8, loose=0.2, normal ori
    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201807041847.h5'
    
    # label mode = mean
    #srcfile = 'source_epochs_allsubs_HCPMMP1_201708232002.h5'
    
    # label mode = mean_flip
    #srcfile = 'source_epochs_allsubs_HCPMMP1_201706131725.h5'
    
    # label mode= None, lselection=[V1, V2], window=[0, 0.4]
    #srcfile = 'source_epochs_allsubs_HCPMMP1_201708151431.h5'
    
    # label mode = (abs) max
    #srcfile = 'source_epochs_allsubs_HCPMMP1_201706271717.h5'
    
    # label mode= None, lselection=pre-motor and motor areas, window=[0.3, 0.9]
    #srcfile = 'source_epochs_allsubs_HCPMMP1_201710231731.h5'
    
    # label mode = None, lselection=premotor, motor, mid cingulate, parietal
    # window = [-0.3, 2.5], no baseline correction before source reconstruction
#    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201802081827.h5'
    
    # loose=0.2, normal ori
    # label mode = None, lselection=premotor, motor, mid cingulate, parietal
    # window = [-0.3, 2.5], no baseline correction before source reconstruction
#    srcfile = 'source_epochs_allsubs_HCPMMP1_5_8_201807051740.h5'
    
    if os.name == 'posix':
        srcfile = os.path.join('mne_subjects', 'fsaverage', 'bem', srcfile)
    else:
        srcfile = os.path.join('E:/', 'bitzer', 'mne_subjects', 'fsaverage', 
                               'bem', srcfile)
    epname = 'label_tc'

# where to store
file = pd.datetime.now().strftime(datatype+'_crosscorr'+'_%Y%m%d%H%M'+'.h5')
# tmp-file should use local directory (preventing issue with remote store)
if os.name == 'posix':
    tmpfile = os.path.join('/media/bitzer/Data', file+'.tmp')
else:
    tmpfile = os.path.join('E:/', 'bitzer', file+'.tmp')
file = os.path.join(helpers.resultsdir, file)

with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
    store['options'] = pd.Series(dict(
            exclude_to=exclude_to, toolate=toolate))
    store['labels'] = pd.Series(labels)
    store['srcfile'] = pd.Series(srcfile)


#%% extract some basic info from example data
subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
S = subjects.size

with pd.HDFStore(srcfile, 'r') as store:
    epochs = store.select(epname, 'subject=2')

# times stored in epochs
epochtimes = epochs.index.levels[2]
# desired times for the analysis
times = epochtimes[slice(*epochtimes.slice_locs(*timeslice))]

# trials available in data (should be the same as in DM)
trials = epochs.index.get_level_values('trial').unique()

# figure out the maximum number of dots that we could consider in the analysis
# based on the available epoch times (absolute maximum is 25 by exp design)
dots = np.arange(1, min(25 + 1, epochtimes[-1] // (helpers.dotdt * 1000) + 2), 
                 dtype=int)

gc.collect()


#%% get subject responses
choices = regressors.subject_trial.response
rts = regressors.subject_trial.RT * 1000


#%% 
dott = lambda t0: t0 + (dots - 1) * int(helpers.dotdt * 1000)

def pearson_cross_correlate(X, Y):
    """Computes the cross-correlation between X and Y.
    
    output[i, j] = scipy.stats.pearsonr(X[:, i], Y[:, j])
    
    the only difference to scipy is that here I compute this more efficiently
    for all columns of Y using array operations
    """
    X = np.array(X)
    Y = np.array(Y)
    
    X = X - X.mean()
    Y = Y - Y.mean()
    sumY2 = np.sum(Y ** 2, axis=0)
    
    return np.concatenate(
            [(np.sum(X[:, i][:, None] * Y, axis=0) 
             / np.sqrt(np.sum(X[:, i] ** 2) * sumY2))[None, :]
            for i in range(X.shape[1])])


#%%
# original sequential order
permutation = np.arange(trials.size * epochtimes.size).reshape(
        [trials.size, epochtimes.size])

for perm in range(nperm+1):
    if perm > 0:
        # randomly permute trials, use different permutations for time points, but use
        # the same permutation across all subjects such that only trials are permuted,
        # but random associations between subjects are maintained
        for t in range(epochtimes.size):
            # shuffles axis=0 inplace
            np.random.shuffle(permutation[:, t])
    
    for sub in subjects:
        startt = datetime.now()
        
        with pd.HDFStore(srcfile, 'r') as store:
            epochs = store.select(epname, 'subject=sub').loc[sub]
        gc.collect()
        
        epochs[:] = epochs.values[permutation.flatten(), :]
        
        # set data too close to RT to nan so it can be later easily excluded
        for tr in trials:
            epochs.loc[(tr, slice(rts.loc[(sub, tr)] + toolate + 1e-6, None)), 
                       epochs.columns[0]] = np.nan
        
        # setup output
        correlations = pd.DataFrame(
                [], dtype=float,
                index=pd.MultiIndex.from_product(
                        [[perm], [sub], times, delays, labels],
                        names=['perm', 'subject', 'seedt', 'delay', 'seed']),
                columns=pd.Index(labels, name='cmp'))
        
        for si, seedt in enumerate(times):
            for di, delay in enumerate(delays):
                print('\rperm %d, subject %2d, seedt %2d of %2d, '
                      'delay %2d of %2d' % (
                              perm, sub, si+1, len(times), di+1, len(delays)), 
                      end='', flush=True)
                
                ts = dott(seedt)
                seed_data = epochs.loc[(slice(None), ts), :].copy()
                seed_data.index = pd.MultiIndex.from_product(
                        [trials, dots[ts <= epochtimes[-1]]], 
                        names=['trial', 'dot'])
                
                ts = dott(seedt + delay)
                cmp_data = epochs.loc[(slice(None), ts), :].copy()
                cmp_data.index = pd.MultiIndex.from_product(
                        [trials, dots[ts <= epochtimes[-1]]], 
                        names=['trial', 'dot'])
                
                data = pd.concat([seed_data, cmp_data], keys=['seed', 'cmp'],
                                 names=['data', 'label'], axis=1)
                
                data.dropna(inplace=True)
                
                corrs = pearson_cross_correlate(
                        data['seed'][labels], data['cmp'][labels])
                correlations.loc[
                        (perm, sub, seedt, delay, slice(None)), :] = corrs
        print('')

        endt = datetime.now()

        with pd.HDFStore(file, mode='a', complib='blosc', complevel=7) as store:
            store.append('correlations', correlations)
            store.append('timing', pd.DataFrame(
                    [[startt, endt, endt-startt]],
                    columns=['start', 'end', 'elapsed'],
                    index=pd.MultiIndex.from_tuples(
                            [(perm, sub)],
                            names=['perm', 'subject'])))
