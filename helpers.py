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
import rtmodels

dotfile = 'batch_dots_pv2.mat'

defaultdir = os.path.join('data', 'meg_behav')
toresponse = np.r_[0, 5.0]
cond = 25
dotdt = 0.1
dotstd = 70.


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
    
    return dotpos


def get_5th_dot_infos(dotpos):
    D = dotpos.shape[0]
    
    # make batch indices: a batch collects all trials with common dot positions
    # except for the 5th dot position
    
    # there were 28 'long trials' for which we modulated 5th dot positions
    batches = np.arange(1, 29)
    
    # there were 6 different dot positions
    batches = np.tile(batches, 6)
    
    # after the 5th dot modulated trials there were 72 "short trials"
    batches = np.r_[batches, np.arange(29, 29+72)]
    
    # then these trials were repeated, but with opposite sign of the x-coordinate
    # conveniently, the batch numbers can restart at 101
    batches = np.r_[batches, 100+batches]
    
    # put into DataFrame with index corresponding to order in dotpos
    trial_info = pd.DataFrame(batches, index=pd.Index(np.arange(480)+1, 
                              name='trial'), columns=['batch'])
    
    # get "correct" decision after 4th dot from ideal observer
    feature_means = np.c_[[-cond, 0], [cond, 0]]
    model = rtmodels.discrete_static_gauss(dt=dotdt, maxrt=dotdt*D, 
                                           toresponse=toresponse, 
                                           choices=[-1, 1], Trials=dotpos, 
                                           means=feature_means)
    
    # set ideal observer parameters
    model.intstd = dotstd
    model.noisestd = 1e-15
    model.prior = 0.5
    
    # generate log-posterior beliefs
    logpost, _ = model.compute_logpost_from_features(np.arange(480))
    
    # get "correct" choice after 4th dot as the choice with the largest posterior
    # belief at that time point; this is the same as summing the x-coordinates of
    # the first 4 dots and checking whether the sum is positive: if yes, correct 
    # choice is 1 else -1
    trial_info['correct_4th'] = model.choices[np.argmax(logpost[3, :, :, 0], 
                                                        axis=0)]
    
    # raw 5th dot position (x-coordinate)
    trial_info['5th_dot'] = dotpos[4, 0, :]
    
    # The penalty is defined in terms of the correct choice according to the first 
    # 4 dot positions, so it makes sense to define the 5th dot position in relation
    # to whether it supports the correct choice, or not. If the correct choice is
    # right (1) a positive dot position (x>0) supports the correct choice and 
    # should lead to a low penalty, but if the correct choice is left (-1), a 
    # positive dot position supports the wrong choice and should lead to a high
    # penalty. By multiplying the 5th dot position with the (sign of) the correct
    # choice we can flip that relationship such that a larger value always 
    # indicates greater support for the correct choice.
    trial_info['support_correct'] = dotpos[4, 0, :] * trial_info['correct_4th']
    
    # bin 5th dot positions
    # the manipulated 5th dot positions were 64 pixels apart and the used values
    # were [-25, 25] + [-160, -96, -32, 32, 96, 160]
    bins = np.r_[-300, np.arange(-160, 200, 64), 300]
    trial_info['support_correct_bin'] = np.digitize(
        trial_info['support_correct'], bins)
    
    return trial_info
    
    
def load_subject_data(subject_index, behavdatadir=defaultdir, cond=cond, 
                      toresponse=toresponse):
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
    
    
def load_all_responses(behavdatadir=defaultdir, cond=cond, 
                       toresponse=toresponse):
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