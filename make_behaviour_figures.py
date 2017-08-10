#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:21:58 2017

@author: bitzer
"""

import regressors
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')


#%% plot across-subject RT distribution
RTs = regressors.subject_trial.RT.copy()

# set timed out RTs to nan
RTs[RTs == 5.0] = np.nan
RTs.dropna(inplace=True)
RTs.sort_values(inplace=True)

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(7.5, 3))
sns.distplot(RTs, ax=axes[0])

axes[0].set_xlabel('response time (s)')
axes[0].set_ylabel('density')

axes[1].plot(RTs.values, (np.arange(RTs.size)+1) / RTs.size)
axes[1].set_xlabel('response time (s)')
axes[1].set_ylabel('cumulative density')

axes[1].set_xlim([RTs.min(), RTs.max()])

fig.subplots_adjust(left=0.1, bottom=0.17, right=0.97, top=0.91, wspace=0.3)

fig.savefig(os.path.join(figdir, 'pooledRTs.png'),
            dpi=300)

