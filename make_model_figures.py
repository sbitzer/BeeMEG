#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:11:25 2018

@author: bitzer
"""

import helpers
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.stats


#%% define model
targets = helpers.cond * np.r_[1, -1]
cov = helpers.dotstd * np.eye(2)

def evidence(dots):
    return (scipy.stats.multivariate_normal.logpdf(
                    dots, mean=np.r_[targets[0], 0], cov=cov) - 
            scipy.stats.multivariate_normal.logpdf(
                    dots, mean=np.r_[targets[1], 0], cov=cov))

#%%
fig, ax = plt.subplots(figsize=(4, 3));

dots = np.c_[np.arange(-200, 200), np.random.randn(400)]
ax.plot(dots[:, 0], evidence(dots), label='x-Koordinate (y beliebig)', lw=2,
        color='#93107d')

dots = np.c_[np.zeros(400), np.arange(-200, 200)]
ax.plot(dots[:, 1], evidence(dots), label='y-Koordinate (x=0)', lw=2,
        color='#717778')

ax.set_xlabel('Position des Punktes (px)')
ax.set_ylabel('Evidenz für rechts')
ax.legend()

fig.tight_layout()

fig.savefig(os.path.join(helpers.figdir, 'dot-position_vs_evidence.png'))