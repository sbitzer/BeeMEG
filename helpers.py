#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:13:55 2016

@author: bitzer
"""

import os
import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd
import re
import rtmodels
import numba
import mne

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
    

def load_meg_epochs_from_sdat(subject, megdatadir=megdatadir):
    """load Hame's epoched data for one subject, put it into MNE epoch container."""
    
    # conveniently, the order of the channels in the matlab data is the same as
    # the order given by the raw fif-file (see channel_names.mat which I extracted
    # from variable hdr in one of the files in meg_original_data)
    mat = loadmat(os.path.join(megdatadir, '%02d_sdat.mat' % subject))

    # construct full data array corresponding to the channels defined in info
    data = np.zeros((480, 313, 301))
    
    # MEG data
    data[:, :306, :] = mat['smeg']
    
    # EOG + ECG
    data[:, 306:309, :] = mat['setc'][:, :3, :]
    
    # triggers (STI101) - Hame must have added this manually as I can't see where
    # it was added in her Matlab scripts, but the time-courses and values 
    # definitely suggest that it's STI101
    # NOT PRESENT FOR ALL SUBJECTS
    if mat['setc'].shape[1] == 4:
        data[:, 310, :] = mat['setc'][:, 3, :]
    
    tmin = -0.3
    dt = 0.004
    
    # load info from raw file
    try:
        # raw file of subject, block 1
        rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm%02da/bm%02da1.fif' % (subject, subject)
        
        # get MEG-info from raw file (for channel layout and names)
        info = mne.io.read_info(rawfile)
    except FileNotFoundError:
        # just use info of subject 2 - shouldn't matter unless you use
        # projections
        rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm02a/bm02a1.fif'
        
        # get MEG-info from raw file (for channel layout and names)
        info = mne.io.read_info(rawfile)
        
    # set sampling frequency to that of Hame's epochs
    info['sfreq'] = 1 / dt
        
    # create events array
    events = np.c_[np.arange(480), np.full(480, 301, dtype=int), 
                   np.zeros(480, dtype=int)]
    
    # type of event
    event_id = {'dot_1_onset': 0}
    
    # create MNE epochs container
    epochs = mne.EpochsArray(data, info, events, tmin, event_id, proj=False)
    
    return epochs


def find_precomputed_epochs(hfreq, sfreq, window, chtype, bl, 
                            megdatadir=megdatadir):
    
    if bl is None:
        blstr = ''
    else:
        blstr = '_bl%.2f-%.2f' % bl
        
    # if there is a file with the exact name, return data from that
    fname = 'epochs_hfreq%.1f_sfreq%.1f_window%.2f-%.2f_%s%s.h5' % (hfreq,
            sfreq, window[0], window[1], chtype, blstr)
    file = os.path.join(megdatadir, fname)
    if os.path.isfile(file):
        return pd.read_hdf(file)
    
    # otherwise check whether there are files for which only the time window
    # differs
    fname_mask = 'epochs_hfreq%.1f_sfreq%.1f_window*_%s%s.h5' % (hfreq,
            sfreq, chtype, blstr)
    files = glob.glob(os.path.join(megdatadir, fname_mask))
    
    # sort files according to their size (smallest first)
    files = pd.DataFrame(files, index=[os.path.getsize(file) for file in files])
    files.sort_index(inplace=True)
    
    # return data from the first file which includes all requested data
    for file in files.values[:, 0]:
        filewin = re.match(r'.*window(-?\d\.\d\d)-(-?\d\.\d\d)', file).groups()
        filewin = [float(filewin[0]), float(filewin[1])]
        
        if filewin[0] <= window[0] and filewin[1] >= window[1]:
            epochs = pd.read_hdf(file)
            
            filetimes = epochs.index.get_level_values('time')
            window = np.array(window) * 1000
            
            if window[0] == window[1]:
                time = filetimes[np.argmin(np.abs(filetimes - window[0]).values)]
                return epochs.xs(time, level='time', drop_level=False)
            else:
                return epochs[np.logical_and(filetimes >= window[0], 
                                             filetimes <= window[1])]
            
    return None

            
def load_meg_epochs(hfreq=10, sfreq=100, window=[0.4, 0.7], chtype='mag', 
                    bl=None, megdatadir=megdatadir, save_evoked=False):
    
    if bl is None:
        blstr = ''
    else:
        blstr = '_bl%.2f-%.2f' % bl
    fname = 'epochs_hfreq%.1f_sfreq%.1f_window%.2f-%.2f_%s%s.h5' % (hfreq,
            sfreq, window[0], window[1], chtype, blstr)
    file = os.path.join(megdatadir, fname)
    
    epochs_all = find_precomputed_epochs(hfreq, sfreq, window, chtype, bl, 
                                         megdatadir)
    if epochs_all is None:
        subjects = find_available_subjects(megdatadir=megdatadir)
        
        # channel connectivity from FieldTrip for cluster analysis
        connectivity, co_channels = mne.channels.read_ch_connectivity(
            '/home/bitzer/proni/BeeMEG/MEG/neuromag306mag_neighb.mat')
        
        fresh = True
        for sub in subjects:
            print('loading subject %2d' % sub)
            # load MNE epochs
            epochs = load_meg_epochs_from_sdat(sub, megdatadir)
            
            # pick specific channels
            epochs = epochs.pick_types(meg=chtype)
            
            # correct for baseline
            if bl is not None:
                epochs = epochs.apply_baseline(bl)
            
            # resample
            if sfreq != epochs.info['sfreq']:
                epochs = epochs.resample(sfreq)
            
            # select time points
            epochs = epochs.crop(*window)
            
            # smooth
            if hfreq < sfreq:
                epochs.savgol_filter(hfreq)
            
            # get in data frame with my index
            df = epochs.to_data_frame()
            df.index = pd.MultiIndex.from_product([[sub], np.arange(480)+1, 
                    df.index.levels[2]], names=['subject', 'trial', 'time'])
            
            if fresh:
                if save_evoked:
                    evoked = epochs.average()
                    evoked.save(os.path.join(megdatadir, 'evoked_'
                            'sfreq%.1f_window%.2f-%.2f_%s-ave.fif' % (
                            sfreq, window[0], window[1], chtype)))
                
                epochs_all = df
                fresh = False
            else:
                epochs_all = pd.concat([epochs_all, df], axis=0)
            
            print('')
            
            epochs_all.to_hdf(file, 'epochs_all', mode='w', complevel=7, complib='zlib')
    
    return epochs_all


def load_evoked_container(sfreq=100, window=[0.4, 0.7], chtype='mag', 
                          megdatadir=megdatadir):
    
    fname = 'evoked_sfreq%.1f_window%.2f-%.2f_%s-ave.fif' % (
            sfreq, window[0], window[1], chtype)
    file = os.path.join(megdatadir, fname)
    
    if os.path.isfile(file):
        return mne.read_evokeds(file, proj=False)[0]
    else:
        sub = 2
        
        # load Hame's epoched data
        epochs = load_meg_epochs_from_sdat(sub, megdatadir)
            
        # pick specific channels
        epochs = epochs.pick_types(meg=chtype)
        
        # resample
        epochs = epochs.resample(sfreq)
        
        # select time points for 5th dot
        epochs = epochs.crop(*window)
        
        evoked = epochs.average()
        
        evoked.save(file)
        
    return evoked
    
    

@numba.jit(nopython=True)
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


@numba.jit((numba.float64[:,:], numba.float64[:,:], 
            numba.optional(numba.int64[:])), nopython=True)
def glm_t(data, DM, selecti):
    """Compute t-values for regressors in a GLM with ordinary least squares.
    
        Computations are taken from Wikipedia_ and cross-checked with 
        statsmodels.
        
        Parameters
        ----------
        data : 2D-array (observations x data sets)
            the data to be fitted with the GLM
            
        DM : 2D-array (observations x regressors)
            the design matrix
            
        selecti : 1D-array of ints, or None
            indeces of regressors for which t-values should be returned
            
        Returns
        -------
        tvals : 2D-array (data sets x number of selected regressors)
            the computed t-values of the selected regressors
        
        .. _Wikipedia: https://en.wikipedia.org/wiki/Ordinary_least_squares
    """
    N, M = data.shape
    P = DM.shape[1]
    
    if selecti is None:
        selecti = np.arange(P)
    O = selecti.size
    selecti = selecti.astype(np.uint64)
    
    DM2inv = np.linalg.inv(np.dot(DM.T, DM))
    DM2inv_diag = np.diag(DM2inv)
    
    tvals = np.zeros((M, O))
    for i in range(M):
        beta = np.dot(np.dot(DM2inv, DM.T), data[:, i])
        resid = data[:, i] - np.dot(DM, beta)
        s2 = np.dot(resid.T, resid) / (N - P)
        sei = np.sqrt(s2 * DM2inv_diag[selecti])
        tvals[i, :] = beta[selecti] / sei
        
    return tvals
