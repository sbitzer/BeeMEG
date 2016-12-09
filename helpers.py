#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:13:55 2016

@author: bitzer
"""

import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import re

dotfile = 'batch_dots_pv2.mat'

defaultdir = os.path.join('data', 'meg_behav')
defaultto = np.r_[0, 5.0]
defaultcond = 25


def load_dots(dotfile=dotfile, dotdir=defaultdir):
    # get dot positions
    mat = loadmat(os.path.join(dotdir, dotfile), 
                  variable_names=['xdots_n', 'ydots_n'])
    
    # L - number of trials, S - maximum number of presented dots in one trial
    L, D = mat['xdots_n'].shape
    # last element is the target position for that trial, so ignore
    D = D - 1
    dotpos = np.full((D, 2, L), np.nan)
    dotpos[:, 0, :] = mat['xdots_n'][:, :-1].T
    dotpos[:, 1, :] = mat['ydots_n'][:, :-1].T
    
    # dot constants
    dotstd = 70.
    dotdt = 0.1
    
    return dotpos, dotstd, dotdt
    

def load_subject_data(subject_index, behavdatadir=defaultdir, cond=defaultcond, 
                      toresponse=defaultto):
    """Load responses of a subject, recode to -1, 0, 1 for left, timed-out, right."""
    
    mat = loadmat(os.path.join(behavdatadir, '%02d-meg.mat' % subject_index), 
                  variable_names=['m_dat'])
    
    extract_data = lambda mat: np.c_[np.abs(mat[:, 0]), np.sign(mat[:, 0]),
        mat[:, 5], mat[:, 3] / 10000]
    
    colnames = ['condition', 'stimulus', 'response', 'RT']
    
    # make a data frame with the relevant data
    respRT = pd.DataFrame(extract_data(mat['m_dat']), 
                          index=np.int16(mat['m_dat'][:, 1]), 
                          columns=colnames)
    respRT.index.name = 'trial'
    
    # set RT of timed out responses to 5s
    respRT['RT'][respRT['response'] == toresponse[0]] = toresponse[1]
    
    # recode timed out responses (=0) as 1.5: this makes them 0 after the next
    # transformation to -1 and 1
    respRT.replace({'response': {toresponse[0]: 1.5}}, inplace=True)
    
    # unify coding of left stimuli (1 = -1) and right stimuli (2 = 1)
    respRT['response'] = (respRT['response'] - 1.5)  * 2    
    
    # there should only be the given condition in the data
    assert(np.all(respRT['condition'] == cond))
    
    return respRT
    
    
def find_available_subjects(behavdatadir=defaultdir):
    _, _, filenames = next(os.walk(behavdatadir))
    subjects = []
    for fname in filenames:
        match = re.match('^(\d+)-meg.mat$', fname)
        if match is not None:
            subjects.append(int(match.group(1)))
    
    return np.sort(np.array(subjects))
    
    
def load_all_responses(behavdatadir=defaultdir, cond=defaultcond, 
                       toresponse=defaultto):
    subjects = find_available_subjects(behavdatadir)
    
    allresp = pd.DataFrame([])
    for sub in subjects:
        respRT = load_subject_data(sub, behavdatadir, cond, toresponse)
        del respRT['condition']
        respRT['subject'] = sub
        
        allresp = pd.concat([allresp, respRT])
        
    allresp['correct'] = allresp['stimulus'] == allresp['response']
    del allresp['stimulus']
    
    allresp.set_index('subject', append=True, inplace=True)
    allresp = allresp.reorder_levels(['subject', 'trial'])
    
    return allresp