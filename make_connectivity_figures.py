#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:20:58 2018

@author: bitzer
"""
import helpers
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import MDS
from scipy.stats import pearsonr


#%% load data
file = 'source_crosscorr_201807131159.h5'

file = os.path.join(helpers.resultsdir, file)

subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)

# subject to investigate, set to sub=0 for mean correlations across all stored
sub = 0

perm = 0
with pd.HDFStore(file, 'r') as store:
    if sub == 0:
        broken = False
        cmean = np.zeros((51 * 41 * 34, 34))
        for cnt, su in enumerate(subjects):
            correlations = store.select('correlations', 
                                        'perm=perm & subject=su')
            if correlations.size == 0:
                broken = True
                su = subjects[cnt-1]
                correlations = store.select('correlations', 
                                            'perm=perm & subject=su')
                break
            else:
                cmean += correlations.values
                
        if not broken:
            cnt += 1
            
        correlations[:] = cmean / cnt
        correlations = correlations.reset_index('subject')
        correlations['subject'] = sub
        correlations.set_index('subject', inplace=True, append=True)
        correlations = correlations.reorder_levels(
                ['perm', 'subject', 'seedt', 'delay', 'seed'])
    else:
        correlations = store.select('correlations', 'perm=perm & subject=sub')
    
    timing = store.timing
    
delays = correlations.loc[(0, sub)].index.get_level_values('delay').unique()
times = correlations.loc[(0, sub)].index.get_level_values('seedt').unique()
imext = [times.min(), times.max(), delays.min(), delays.max()]

full = lambda la, hemi: '%s_%s_ROI-%sh' % (hemi, la, hemi.lower())

labels = correlations.columns

    
#%% show correlation plates for two areas for all 4 combinations of hemisphere
seed = 'v23ab'
cmp = '6a'

#lims = [-1, 1]
lims = correlations.loc[(
        perm, sub, slice(None), slice(None), 
        [full(seed, 'L'), full(seed, 'R')]), [full(cmp, 'L'), full(cmp, 'R')]]
lims = [-lims.abs().values.max(), lims.abs().values.max()]

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

ims = np.zeros((2, 2), dtype=object)
for i, (arow, seedh) in enumerate(zip(axes, ['L', 'R'])):
    for j, (ax, cmph) in enumerate(zip(arow, ['L', 'R'])):
        corrs = correlations.loc[
                (perm, sub, slice(None), slice(None), full(seed, seedh)),
                full(cmp, cmph)].unstack('seedt')
        
        ims[i, j] = ax.imshow(corrs, origin='lower', extent=imext, 
                              vmin=lims[0], vmax=lims[1], cmap='PiYG')
        ax.set_title('{0} -> {1}'.format(seedh, cmph))

for i in range(2):
    axes[i, 0].set_ylabel('delay for correlation (ms)')
    axes[1, i].set_xlabel('time after dot onset (ms)')

cax = fig.add_axes((0.9, 0.25, 0.02, 0.5))
fig.colorbar(ims[0, 1], cax=cax)

fig.text(0.5, 0.94, '%s -> %s' % (seed, cmp), ha='center', va='center', 
         size='x-large')


#%% show correlation for one particular connection and seed time across delays
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

seed = full('4', 'L')
cmp = full('p24pr', 'L')
seedt = 180
delay = 0

corrs = correlations.loc[(perm, sub, seedt, slice(None), seed), cmp]

axes[0].plot(corrs.index.get_level_values('delay'), corrs)
axes[0].set_xlabel('delay for correlation (ms)')
axes[0].set_ylabel('correlation coefficient')
axes[0].set_title('%s -> %s at seedt=%d ms' % (seed[:-7], cmp[:-7], seedt))

corrs = correlations.loc[(perm, sub, slice(None), delay, seed), cmp]

axes[1].plot(corrs.index.get_level_values('seedt'), corrs)
axes[1].set_xlabel('seed time (ms)')
axes[1].set_title('%s -> %s at delay=%d ms' % (seed[:-7], cmp[:-7], delay))

fig.tight_layout()


#%% compute some summary measures
perm = 0
seedt = 180
delay = 0

# the delay value which has largest correlation magnitude at seedt
maxcorr = pd.DataFrame(np.zeros((len(labels), len(labels))), dtype=int,
                       index=labels, columns=labels)
# average correlation magnitude at seedt across delays
avmagcorr = pd.DataFrame(np.zeros((len(labels), len(labels))), dtype=float,
                         index=labels, columns=labels)
# average correlation magnitude at delay across seed times
avmagcorrdelay = pd.DataFrame(
        np.zeros((len(labels), len(labels))), dtype=float, index=labels, 
        columns=labels)

for label in labels:
    corrs = correlations.loc[(perm, sub, seedt, slice(None), label)].abs()
    maxcorr.loc[label, :] = corrs.idxmax()
    avmagcorr.loc[label, :] = corrs.mean()
    
    corrs = correlations.loc[(perm, sub, slice(None), delay, label)].abs()
    avmagcorrdelay.loc[label, :] = corrs.mean()


#%% compare avmagcorr and avmagcorrdelay
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

axes[0].matshow(avmagcorr / avmagcorr.values.sum())
axes[1].matshow(avmagcorrdelay / avmagcorrdelay.values.sum())


#%% visualise the relationship defined by these measures using MDS
dissim = 1 - avmagcorrdelay.values
dissim = (dissim + dissim.T) / 2

mds = MDS(n_components=3, max_iter=3000, eps=1e-9,
          dissimilarity="precomputed", n_jobs=1)
embedding = mds.fit(dissim).embedding_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=['k' if l.startswith('L') else 'r' for l in labels])


#%% simulate oscillatory signal
st = np.linspace(0, 6*np.pi, 30)
de = np.linspace(-np.pi, np.pi, 7)

corrs = pd.DataFrame([], dtype=float, index=pd.Index(st, name='seedt'),
                     columns=pd.Index(de, name='delay'))

ti = np.linspace(0, 12*np.pi, 60)

for s in st:
    for d in de:
        corrs.loc[s, d], _ = pearsonr(
                np.sin(ti + s), np.sin(ti + s + d))
        
corrs.plot()