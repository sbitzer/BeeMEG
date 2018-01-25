#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:16:54 2017

@author: bitzer
"""

import helpers
import pandas as pd
import os
import matplotlib.pyplot as plt

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% load data

# baseline = (-0.3, 0)
#fname = 'meg_singledot_201706131051.h5'

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
fname = 'source_singledot_201801171913.h5'

resfile = os.path.join(helpers.resultsdir, fname)

evoked = helpers.load_evoked_container(window=[0, 0.9])

second_level = pd.read_hdf(resfile, 'second_level')

perms = second_level.index.levels[0]

dots = pd.read_hdf(resfile, 'dots')


#%% plot example signal
r_name = 'dot_x'
nperm = 3
regs = ['{}_{}'.format(r_name, d) for d in dots]
R = len(regs)

measure = 'mean'
#sensor = 'MEG0741'
sensor = 'L_4_ROI-lh'

fig, axes = plt.subplots(R, 1, sharex=True, sharey=True, figsize=[6, 8])
axes[0].set_title('x-coordinate (%s)' % sensor)

times = second_level.index.levels[2]
for r, reg in enumerate(regs):
    line, = axes[r].plot(
            times, second_level.loc[(0, sensor, slice(None)), (measure, reg)])
    for p in range(1, nperm+1):
        pl, = axes[r].plot(
                times, 
                second_level.loc[(p, sensor, slice(None)), (measure, reg)],
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

axes[-1].legend([line, pl, onsetl], ['data', 'permuted', 'dot onset'])
axes[-1].set_xlabel('time (ms)')

fig.subplots_adjust(left=0.1, bottom=0.06, right=0.95, top=0.96, hspace=0.18)

#fig.savefig(os.path.join(figdir, 'singledot_%s_%s.png' % (r_name, sensor)), 
#            dpi=300)