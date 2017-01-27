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
from numba import jit

dotfile = 'batch_dots_pv2.mat'

# directories
basedir = 'data'
behavdatadir = os.path.join(basedir, 'meg_behav')
megdatadir = os.path.join(basedir, 'meg_final_data')
resultsdir = os.path.join(basedir, 'inf_results')

toresponse = np.r_[0, 5.0]
cond = 25
dotdt = 0.1
dotstd = 70.


def load_dots(dotfile=dotfile, dotdir=basedir):
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


def get_ideal_observer(dotpos=None, bound=0.8):
    if dotpos is None:
        dotpos = load_dots()
    D = dotpos.shape[0]
    
    feature_means = np.c_[[-cond, 0], [cond, 0]]
    model = rtmodels.discrete_static_gauss(dt=dotdt, maxrt=dotdt*D, 
                                           toresponse=toresponse, 
                                           choices=[-1, 1], Trials=dotpos, 
                                           means=feature_means)
    
    # set ideal observer parameters
    model.intstd = dotstd
    model.noisestd = 1e-15
    model.prior = 0.5
    model.lapseprob = 0.0
    model.ndtmean = -100
    model.bound = bound
    
    return model


def get_5th_dot_infos(dotpos):
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
    model = get_ideal_observer(dotpos)
    
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
    trial_info['support_correct_4th'] = dotpos[4, 0, :] * trial_info['correct_4th']
    
    # same thing, but based on 'true correct'
    trial_info['support_correct'] = dotpos[4, 0, :] * np.sign(
        dotpos[:, 0, :].mean(axis=0))
    
    # bin 5th dot positions
    # the manipulated 5th dot positions were 64 pixels apart and the used values
    # were [-25, 25] + [-160, -96, -32, 32, 96, 160]
    bins = np.r_[-300, np.arange(-160, 200, 64), 300]
    trial_info['support_correct_bin_4th'] = np.digitize(
        trial_info['support_correct_4th'], bins)
    trial_info['support_correct_bin'] = np.digitize(
        trial_info['support_correct'], bins)
    
    return trial_info
    
    
def load_subject_data(subject_index, behavdatadir=behavdatadir, cond=cond, 
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
    
    
def find_available_subjects(behavdatadir=None, megdatadir=None):
    """checks directories for data files (one per subject) and returns available subjects."""
    
    if behavdatadir is not None:
        _, _, filenames = next(os.walk(behavdatadir))
        subjects_behav = []
        for fname in filenames:
            match = re.match('^(\d+)-meg.mat$', fname)
            if match is not None:
                subjects_behav.append(int(match.group(1)))
    if megdatadir is not None:
        _, _, filenames = next(os.walk(megdatadir))
        subjects_meg = []
        for fname in filenames:
            match = re.match('^(\d+)_sdat.mat$', fname)
            if match is not None:
                subjects_meg.append(int(match.group(1)))
                
    if megdatadir is None and behavdatadir is None:
        raise ValueError('please provide at least one data directory')
    elif megdatadir is not None and behavdatadir is not None:
        return np.intersect1d(subjects_meg, subjects_behav)
    elif megdatadir is not None:
        return np.sort(subjects_meg)
    else:
        return np.sort(subjects_behav)
    
    
def load_all_responses(behavdatadir=behavdatadir, cond=cond, 
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
    

@jit(nopython=True)
def linregress_t(data, predictors):
    """computes t-values for the slope of data = slope*predictor + intercept
    
    The computations are taken from scipy.stats.linregress and cross-checked 
    with ordinary least squares of statsmodels.
    
    Parameters
    ----------
    data : 2D-array (observations x locations)
        the data which should be predicted, t-values will be computed 
        independently for each of the given locations
    predictors: 1D-array (observations)
        predictor values which are supposed to predict the data
        
    Returns
    -------
    tvals : 1D-array (locations)
        t-values computed for the data at each location
    """
    # numba doesn't recognise the condition on ndim and compilation fails with
    # an error, because it thinks that data will become 3D for a given 2D array
#    if data.ndim == 1:
#        data = np.expand_dims(data, 1)
    N, M = data.shape
    
    TINY = 1.0e-20
    df = N - 2
    
    ssxm = np.var(predictors)
    xm = predictors - predictors.mean()
    
    tvals = np.zeros(M)
    for i in range(M):
        ssym = np.var(data[:, i])
        r_num = np.dot(xm, data[:, i] - data[:, i].mean()) / N
        r_den = np.sqrt(ssxm * ssym)
        if r_den == 0.0:
            r = 0.0
        else:
            r = r_num / r_den
            # test for numerical error propagation
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0
        
        tvals[i] = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
    
    return tvals