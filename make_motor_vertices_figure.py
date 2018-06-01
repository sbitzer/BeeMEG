#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 18:26:44 2018

@author: bitzer
"""

import matplotlib.pyplot as plt
import helpers
import os

hemi = 'lh'

regfig = os.path.join(
        helpers.figdir, 'motor_vertices_dot_x_tval_%s_490_exclate.png' % hemi)
evfig = os.path.join(
        helpers.figdir, 'motor_vertices_evoked_tval_%s_30_evoked.png' % hemi)


#%%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(plt.imread(evfig))
axes[1].imshow(plt.imread(regfig))

axes[0].set_position([0, 0, 0.5, 0.9], 'original')
axes[1].set_position([0.5, 0, 0.5, 0.9], 'original')

for ax in axes:
    ax.set_axis_off()
    
fig.text(0.25, 0.93, "response-aligned grand average", ha='center', 
         fontsize=16)
fig.text(0.75, 0.93, r"dot onset aligned evidence correlation", 
         ha='center', fontsize=16)

fig.text(0.5, 0.8, "p < 0.01\nFDR corrected", 
         ha='center', fontsize=12, color='#909090')

fig.savefig(os.path.join(helpers.figdir, 
                         'motor_vertices_comparison_%s.png' % hemi))