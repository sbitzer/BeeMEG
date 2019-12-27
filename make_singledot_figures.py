#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:16:54 2017

@author: bitzer
"""

import helpers
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


#%% load data

# baseline = (-0.3, 0)
#fname = 'meg_singledot_201706131051.h5'

# magnetometers, baseline = (-0.3, 0), trial normalisation of data, 
# dots 1-6, local normalisation of DM, exclude timeouts, toolate=-200
# 'trial_time', 'dot_x', 'dot_y', 'response', 'intercept'
fname = 'meg_singledot_201912261017.h5'

# magnetometers, baseline = (-0.3, 0), trial normalisation of data, 
# dots 1-6, local normalisation of DM, exclude timeouts, toolate=-200
# 'trial_time', 'dot_x', 'dot_y', 'response', 'entropy', 'intercept'
#fname = 'meg_singledot_201803271303.h5'

# label mode = mean, baseline (-0.3, 0), dots 3-5, 
# source GLM, sum_dot_y_prev_3, accev_3, dot_x, dot_y, trial regressors
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#fname = 'source_singledot_201709071126.h5'

# label mode = mean, baseline (-0.3, 0), dots 1-5, 
# source GLM, dot_x, dot_y, trial regressors
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#fname = 'source_singledot_201711131931.h5'

# label mode = mean, baseline (-0.3, 0), dots 1-7, times [0, 1300]
# source GLM, dot_x, dot_y, trial regressors
# subject-specific normalisation of DM without centering and scaling by std
# label_tc normalised across trials, times and subjects
#fname = 'source_singledot_201801171913.h5'

resfile = os.path.join(helpers.resultsdir, fname)

evoked = helpers.load_evoked_container(window=[0, 0.9])

second_level = pd.read_hdf(resfile, 'second_level')

perms = second_level.index.levels[0]

dots = pd.read_hdf(resfile, 'dots')


#%% determine label with strongest / most consistent correlations
r_name = 'dot_x'
regs = ['{}_{}'.format(r_name, d) for d in dots]
R = len(regs)

avabst = second_level.loc[0, ('tval', regs)].abs().mean(level='label').mean(
        axis=1)

maxlabel = avabst.idxmax()


#%% plot topography of average t-val magnitudes
evoked.data[:, 0] = avabst.values

fig = evoked.plot_topomap(times=[0], vmin=avabst.min(), vmax=avabst.max(),
                          scalings={'mag': 1}, units='t', time_format='',
                          title='evidence\n(average magnitude)',
                          mask=np.tile((avabst.index == maxlabel)[:, None], 
                                       (1, evoked.data.shape[1])),
                          mask_params={'markersize': 10});

fig.savefig(os.path.join(helpers.figdir, 'singledot_mean_abs_t_%s' % r_name),
            dpi=300)


#%% plot example signal
nperm = 3

measure = 'mean'
label = maxlabel
#label = 'MEG2241'
#label = 'MEG0741'
#label = 'L_4_ROI-lh'

fig, axes = plt.subplots(R, 1, sharex=True, sharey=True, figsize=[6, 8])
axes[0].set_title('evidence (%s)' % label)

times = second_level.index.levels[2]
for r, reg in enumerate(regs):
    line, = axes[r].plot(
            times, second_level.loc[(0, label, slice(None)), (measure, reg)])
    for p in range(1, nperm+1):
        pl, = axes[r].plot(
                times, 
                second_level.loc[(p, label, slice(None)), (measure, reg)],
                ':', color=line.get_color(), lw=1)
    axes[r].set_ylabel('dot %s' % reg[-1])
    
    xl = axes[r].get_xlim()
    axes[r].plot(xl, [0, 0], '--k', zorder=1)
    axes[r].set_xlim(xl)

yl = axes[0].get_ylim()
for ax, reg in zip(axes, regs):
    onsetl, = ax.plot(pd.np.ones(2) * (int(reg[-1]) - 1) * 100, yl, ':k', 
                      zorder=1, label='dot onset')
axes[0].set_ylim(yl)

axes[-1].legend([line, pl, onsetl], ['data', 'permuted', 'dot onset'],
                loc='upper left')
axes[-1].set_xlabel('time (ms)')

fig.tight_layout()

fig.savefig(os.path.join(helpers.figdir, 'singledot_%s_%s.png' % (r_name, label)), 
            dpi=300)