#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:36:06 2018

@author: bitzer
"""

import helpers
import source_statistics as ss
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import ttest_1samp


#%% load data
r_name = 'evoked'

# evoked sources -1000 to 1000 after response
basefile = 'evoked_source_201802271701.h5'

fl = pd.read_hdf(os.path.join(helpers.resultsdir, basefile), 'first_level')

# only use unpermuted data
fl = fl.loc[0]

# remove uninformative column labels
fl.columns = fl.columns.levels[0]


#%% compute grand average within selected area
area = '4'

labels = []
evoked = []
for hemi in ['L', 'R']:
    labels.append(hemi + '_' + area)
    ind = fl.index.get_level_values('label').map(lambda x: x.startswith(
            labels[-1]))
    
    evoked.append(fl[ind].groupby('time').mean())

evoked = pd.concat(evoked, keys=labels, names=['label', 'time'])

evoked_sl = pd.DataFrame(evoked.mean(axis=1))
evoked_sl.columns = ['mean']
evoked_sl['ste'] = evoked.std(axis=1) / pd.np.sqrt(evoked.shape[1])
evoked_sl['top'] = evoked_sl['mean'] + 2 * evoked_sl.ste
evoked_sl['bottom'] = evoked_sl['mean'] - 2 * evoked_sl.ste
tvals, pvals = ttest_1samp(evoked, 0, axis=1)
evoked_sl['tval'] = tvals
evoked_sl['mlog10p'] = -pd.np.log10(pvals)
ss.add_measure(evoked_sl, 'p_fdr')

print('largest absolute average t-values:')
print(evoked_sl.groupby('time').mean().abs().tval.sort_values().tail())


#%% plot time course
fig, ax = plt.subplots()

lr = dict(L = 'left', R = 'right')
sigy = dict(L = -0.025, R = -0.027)
cols = dict(L = 'C0', R = 'C1')

for label in labels:
    sl = evoked_sl.loc[label]
    
    ax.plot(sl.index, sl['mean'], label=lr[label[0]], color=cols[label[0]])
    
    ax.fill_between(sl.index, sl.top, sl.bottom, alpha=0.2, 
                    color=cols[label[0]])
    
    ind = sl.p_fdr < 0.01
    ax.plot(sl.index[ind], sigy[label[0]] * pd.np.ones(ind.sum()), '.', 
            color=cols[label[0]])
    
ax.legend()
ax.set_title('area %s' % area)
ax.set_xlabel('time from response (ms)')
ax.set_ylabel('source grand average (au)')
ax.plot(sl.index, pd.np.zeros_like(sl.index), ':k', zorder=-1)

fig.tight_layout()
fig.savefig(os.path.join(
        helpers.figdir, 'grand_average_evoked_%s.png' % area), dpi=300)