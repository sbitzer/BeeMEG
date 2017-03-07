#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:37:05 2017

@author: bitzer
"""

import os
import glob
import re
import pandas as pd


rawdir = '/home/bitzer/proni/BeeMEG/MEG/Raw'

fiffiles = glob.glob(os.path.join(rawdir, '*/*.fif'))

file_re = re.compile(r'.*bm(\d\d)a/bm\1a([\dbd])(_mc)?.fif')

fids = []
for f in fiffiles:
    gr = file_re.match(f).groups()
    if gr[2] is None:
        fids.append([int(gr[0]), gr[1], False, f])
    else:
        fids.append([int(gr[0]), gr[1], True, f])
        
df = pd.DataFrame(fids, columns=['subject', 'part', 'mc', 'fname'])
df.set_index(['subject', 'mc', 'part'], inplace=True)
df.sort_index(inplace=True)

nfiles = pd.Series([df.loc[s].size for s in df.index.levels[0]], 
                   index=df.index.levels[0])

for s in nfiles.index[nfiles < 12]:
    print(df.loc[s])