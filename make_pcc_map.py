#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:14:19 2018

@author: bitzer
"""

import os
import source_visualisations as sv
from surfer import Brain
import mne

figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')

hemi = 'rh'
parc = 'HCPMMP1_5_8'

views = {'rh': ({'azimuth': -146.2, 'elevation': 76.8, 
                 'focalpoint': (-4.3, -5.0, 0.0)}, 64.7, 300.),
         'lh': ({'azimuth': 168.8, 'elevation': 60.3, 
                 'focalpoint': (-0.60, 0., 0.)}, 90, 295.)}

brain = Brain('fsaverage', hemi, 'inflated', cortex='low_contrast',
              subjects_dir=sv.subjects_dir, background='w', 
              foreground='k')

labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi=hemi)

pcc = sv.Glasser_areas[sv.Glasser_areas['main section'] == 18]['area name']
pcc = list(pcc.map(lambda s: hemi[0].upper() + '_' + s + '_ROI-' + hemi))

for label in labels:
    if label.name in pcc:
        brain.add_label(label, borders=1, hemi=hemi, alpha=0.6, 
                        color='k')
        
brain.show_view(*views[hemi])

brain.save_image(os.path.join(figdir, 'pcc_map.png'), antialiased=True)