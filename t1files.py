#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:42:35 2017

@author: bitzer
"""

import os
import pandas as pd
import re
import numpy as np


#%%
sdir = '/media/bitzer/Data/t1scans_beeMEG'

scans = []
skip = True
for dirpath, dirs, files in os.walk(sdir):
    if skip:
        skip = False
        continue
    
    _, subject = os.path.split(dirpath)
    
    for file in files:
        if file.endswith('.txt'):
            file_l = file.lower()
            sequence = None
            if 'mp2rage' in file_l:
                sequence = 'MP2RAGE'
            elif 'mprage' in file_l or 'mpr' in file_l:
                sequence = 'MPRAGE'
            elif 'flash' in file_l:
                sequence = 'Flash'

            if sequence is not None:
                info = re.match(r'.*_([a-zA-Z0-9].*).txt$', file).group(1)
                with open(os.path.join(dirpath, file), errors='ignore') as f:
                    for line in f:
                        if line.startswith('sequenceStart'):
                            m = re.match(r'[\w\s]*:(.*)\(timestamp\)$', line)
                            try:
                                dt = pd.datetime.strptime(
                                        m.group(1), '%Y-%b-%d %H:%M:%S.%f')
                            except ValueError:
                                dt = pd.datetime.strptime(
                                        m.group(1), '%Y-%b-%d %H:%M:%S')
                            break
                
                scans.append([subject, sequence, dt, info, file[:-4]])
                
scans = pd.DataFrame(
        scans, columns=['subject', 'sequence', 'tstamp', 'info', 'file'])

scans.set_index(['subject', 'tstamp'], inplace=True)
scans.sort_index(inplace=True)

#%%
subject_info = pd.read_csv('/home/bitzer/proni/BeeMEG/MEG/subject_data/'
                           'VP_lst_BeeMEG_Hame.csv', header=0, parse_dates=[8],
                           dayfirst=True, index_col=0, skipinitialspace=True)

for sub in subject_info.index:
    sid = subject_info.loc[sub, 'ID']
    t1s = scans.xs(sid, level='subject').copy()
    
    if t1s.shape[0] > 1:
        t1s.loc[:, 'dt'] = np.abs(t1s.index - subject_info.at[sub, 'Datum'])
    else:
        t1s.loc[:, 'dt'] = np.abs(t1s.index - subject_info.at[sub, 'Datum'])[0]
        
    t1s.sort_values('dt', inplace=True)
    
    ind = 0
    t1 = t1s.iloc[0]
    while t1.sequence not in ['MPRAGE', 'MP2RAGE']:
        ind += 1
        if ind < t1s.shape[0]:
            t1 = t1s.iloc[ind]
        else:
            break
        
    subject_info.loc[sub, 'best_t1'] = t1.file
    subject_info.loc[sub, 't1_dt'] = t1['dt']