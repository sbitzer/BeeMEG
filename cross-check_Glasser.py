#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:48:12 2017

@author: bitzer
"""

import pandas as pd
import numpy as np

Glasser_sections = pd.read_csv('Glasser2016_sections.csv', index_col=0)
Glasser_areas = pd.read_csv('Glasser2016_areas.csv', index_col=0)
Glasser_sa = pd.read_csv('Glasser2016_sections_areas.csv')
Glasser_sa.columns = Glasser_sa.columns.map(int)

for section in Glasser_sections.index:
    text = Glasser_sa.loc[:, section].dropna().values
    table = Glasser_areas[Glasser_areas['main section'] == section]['area name'].values
    common = np.intersect1d(table, text)
    
    if (len(table) != len(common)) or (len(text) != len(common)):
        print('inconsistency in section %d' % section)