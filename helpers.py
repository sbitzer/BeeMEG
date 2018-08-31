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
import mne
import gc
from six import print_

try:
    import numba
except:
    pass
try:
    import rtmodels
except:
    pass

dotfile = 'batch_dots_pv2.mat'

# directories
# tmp-directory should use local directory (preventing issue with remote store)
if os.name == 'posix':
    figdir = os.path.expanduser('~/ZIH/texts/BeeMEG/figures')
    basedir = 'data'
    tmpdir = '/media/bitzer/Data'
    bemdir = os.path.join('mne_subjects', 'fsaverage', 'bem')
else:
    figdir = '.'
    basedir = 'X:\BeeMEG\Variables'
    tmpdir = os.path.join('E:/', 'bitzer')
    bemdir = os.path.join('E:/', 'bitzer', 'mne_subjects', 'fsaverage', 'bem')
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


if 'rtmodels' in locals():
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
    mat = loadmat(os.path.join(megdatadir, '%02d_sdatr.mat' % subject))

    # construct full data array corresponding to the channels defined in info
    data = np.zeros((480, 322, 701))
    
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
        rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm%02da/bm%02da1_mc.fif' % (subject, subject)
        
        # get MEG-info from raw file (for channel layout and names)
        info = mne.io.read_info(rawfile)
    except FileNotFoundError:
        # just use info of subject 2 - shouldn't matter unless you use
        # projections
        rawfile = '/home/bitzer/proni/BeeMEG/MEG/Raw/bm02a/bm02a1_mc.fif'
        
        # get MEG-info from raw file (for channel layout and names)
        info = mne.io.read_info(rawfile)
        
    # set sampling frequency to that of Hame's epochs
    info['sfreq'] = 1 / dt
        
    # create events array
    events = np.c_[np.arange(480), np.full(480, 701, dtype=int), 
                   np.zeros(480, dtype=int)]
    
    # type of event
    event_id = {'dot_1_onset': 0}
    
    # create MNE epochs container
    epochs = mne.EpochsArray(data, info, events, tmin, event_id, proj=False)
    
    return epochs


def chtype_str(megtype, **type_kws):
    chstr = ''
    
    if megtype:
        if megtype == True:
            chstr += 'meg'
        else:
            chstr += megtype
        
    for k, v in type_kws.items():
        if v:
            chstr += k
            
    return chstr


def filt_str(filt_freqs, bpfilter=True):
    if bpfilter:
        filtstr = '_filt'
        
        frstrs = []
        for freq in filt_freqs:
            if freq is None:
                frstrs.append('None')
            else:
                frstrs.append('%04.1f' % freq)
                
        filtstr += '-'.join(frstrs)
    else:
        filtstr = ''
        
    return filtstr
    

def find_precomputed_epochs(hfreq, sfreq, window, chstr, bl, filtstr, 
                            megdatadir=megdatadir):
    
    if bl is None:
        blstr = ''
    else:
        blstr = '_bl%.2f-%.2f' % bl
        
    # if there is a file with the exact name, return data from that
    fname = 'epochs_hfreq%.1f_sfreq%.1f_window%.2f-%.2f_%s%s%s.h5' % (hfreq,
            sfreq, window[0], window[1], chstr, blstr, filtstr)
    file = os.path.join(megdatadir, fname)
    if os.path.isfile(file):
        return file
    
    # otherwise check whether there are files for which only the time window
    # differs
    fname_mask = 'epochs_hfreq%.1f_sfreq%.1f_window*_%s%s%s.h5' % (hfreq,
            sfreq, chstr, blstr, filtstr)
    files = glob.glob(os.path.join(megdatadir, fname_mask))
    
    # sort files according to their size (smallest first)
    files = pd.DataFrame(files, index=[os.path.getsize(file) for file in files])
    files.sort_index(inplace=True)
    
    # return data from the first file which includes all requested data
    if files.size > 0:
        for file in files.values[:, 0]:
            filewin = re.match(r'.*window(-?\d\.\d\d)-(-?\d\.\d\d)', 
                               file).groups()
            filewin = [float(filewin[0]), float(filewin[1])]
            
            if filewin[0] <= window[0] and filewin[1] >= window[1]:
                return file
            
    return None


def create_meg_epochs_hdf(hfreq=10, sfreq=100, window=[0, 2.5], chtype='mag',
                          bl=None, megdatadir=megdatadir, save_evoked=False,
                          force_create=False, filt_freqs=None, **type_kws):
    
    chstr = chtype_str(chtype, **type_kws)
    
    if filt_freqs is None:
        bpfilter = False
    else:
        bpfilter = True
        filt_freqs = np.atleast_1d(filt_freqs)
        if filt_freqs.size == 1:
            filt_freqs = np.array([filt_freqs[0], None])
    
    filtstr = filt_str(filt_freqs, bpfilter)
    
    if force_create:
        file = None
    else:
        file = find_precomputed_epochs(hfreq, sfreq, window, chstr, bl, 
                                       filtstr, megdatadir)
    
    if file is None:
        fresh = True
        
        if bl is None:
            blstr = ''
        else:
            blstr = '_bl%.2f-%.2f' % bl
        fname = 'epochs_hfreq%.1f_sfreq%.1f_window%.2f-%.2f_%s%s%s.h5' % (hfreq,
                sfreq, window[0], window[1], chstr, blstr, filtstr)
        file = os.path.join(megdatadir, fname)
        
        with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
            store['scalar_params'] = pd.Series(
                    [hfreq, sfreq], ['hfreq', 'sfreq'])
            store['str_params'] = pd.Series([chtype], ['chtype'])
            store['bl'] = pd.Series(bl)
            store['window'] = pd.Series(window)
            
        subjects = find_available_subjects(megdatadir=megdatadir)
        
        for sub in subjects:
            print_('\ncreating data for subject %2d' % sub, flush=True)
            print_('----------------------------', flush=True)
        
            # load MNE epochs
            epochs = load_meg_epochs_from_sdat(sub, megdatadir)
            
            # pick specific channels
            epochs = epochs.pick_types(meg=chtype, **type_kws)
            
            # filter, if desired
            if bpfilter:
                epochs = epochs.filter(
                        filt_freqs[0], filt_freqs[1], 
                        picks=np.arange(len(epochs.ch_names)),
                        fir_design='firwin')
            
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
            epochdf = epochs.to_data_frame()
            epochdf.index = pd.MultiIndex.from_product(
                    [[sub], np.arange(480)+1, epochdf.index.levels[2]], 
                    names=['subject', 'trial', 'time'])
        
            with pd.HDFStore(file, mode='a', complib='blosc', 
                             complevel=7) as store:
                store.append('epochs', epochdf)
                
            if fresh:
                mean = epochdf.mean()
            else:
                mean += epochdf.mean()
        
        mean = mean / len(subjects)

        with pd.HDFStore(file, 'r+') as store:
            sub = subjects[0]
            std = ((store.select('epochs', 'subject==sub') - mean)**2).sum()
            for sub in subjects[1:]:
                std += ((store.select('epochs', 'subject==sub') - mean)**2).sum()
                
            std = np.sqrt(std / (store.get_storer('epochs').nrows - 1.5))
            
            store['epochs_mean'] = mean
            store['epochs_std'] = std
        
        if save_evoked:
            evoked = epochs.average()
            evoked.save(os.path.join(
                    megdatadir, 
                    'evoked_sfreq%.1f_window%.2f-%.2f_%s-ave.fif' % (
                            sfreq, window[0], window[1], chstr)))
    
    return file


def load_meg_epochs(hfreq=10, sfreq=100, window=[0.4, 0.7], chtype='mag', 
                    bl=None, megdatadir=megdatadir, save_evoked=False):
    
    file = create_meg_epochs_hdf(hfreq, sfreq, window, chtype, bl, megdatadir, 
                                 save_evoked)
    
    epochs = pd.read_hdf(file, 'epochs')
    epochs = epochs.loc[
            (slice(None), slice(None), 
             slice(int(window[0] * 1000), int(window[1] * 1000))), :]
    
    return epochs


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
    
    
if 'numba' in locals():
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


def to_resp_aligned_index(tindex, rts, in_units='ms'):
    """Transforms a first dot onset aligned time index to one aligned at response.
    
    tindex - MultiIndex with levels 'subject', 'trial' and 'time' where 'time'
             is first dot onset aligned in ms with resolution of 10 ms and
             there is the same number of times for each trial of each subject
    rts - Series of response times in ms with index of 'subject' and 'trial'
          with arbitrary resolution
    """
    if in_units == 's':
        tfactor = 1000
    else:
        tfactor = 1
        
    rts = rts * tfactor
    
    # add small jitter to rts that are (almost) exact multiples of 5, because I
    # will later compute round(t-rt, -1) and the round in numpy will then round
    # to the nearest even decade, i.e., it will round both 15 and 25 to 20 
    # which will make consecutive times in the tindex get the same time in the 
    # new rtindex which I want to avoid
    rem10 = np.mod(rts, 10)
    rem5 = np.mod(rts, 5)
    fives = (np.logical_not(np.isclose(rem10, 10) | np.isclose(rem10, 0)) 
             & (np.isclose(rem5, 5) | np.isclose(rem5, 0)))
    rts[fives] = rts[fives] - 1e-7
    
    subject = []
    trial = []
    time = []
    for sub in tindex.get_level_values('subject').unique():
        subind = tindex[slice(*tindex.slice_locs(sub, sub))]
        
        subject.append(np.full(subind.size, sub, dtype=int))
        trials = subind.get_level_values('trial').unique()
        
        T = subind[slice(*subind.slice_locs(
                (sub, trials[0]), (sub, trials[0])))].size
        
        trial.append(np.repeat(trials, T))
        
        time.append(
                np.around(subind.get_level_values('time') * tfactor
                          - np.repeat(rts.loc[(sub, list(trials))], T), 
                          -1).astype(int))
        
    return pd.MultiIndex.from_arrays([
            np.concatenate(subject), 
            np.concatenate(trial),
            np.concatenate(time)], names=('subject', 'trial', 'time'))
    

def load_area_tcs(srcfile, subjects, labels, resp_align=False, rts=None, 
                  normalise=None, exclude_to=False):
    
    if resp_align:
        exclude_to = True
    
    #% example data to get the times that are available
    with pd.HDFStore(srcfile, 'r') as store:
        epochs = store.select('label_tc', 'subject=2')
    epochtimes = epochs.index.levels[2]
    
    #% load data of all subjects
    def load_subject(sub):
        print_('\rloading subject %02d ...' % sub, flush=True, end='')
        with pd.HDFStore(srcfile, 'r') as store:
            epochs = store.select('label_tc', 'subject=sub')[labels]
        
        if exclude_to:
            epochs = epochs.loc[
                    (sub, rts.loc[sub][rts.loc[sub] != toresponse[1] * 1000]
                     .index.values, slice(None))]
        
        if resp_align:
            # get response-aligned times of data points
            epochs.index = to_resp_aligned_index(epochs.index, rts)
        
        # to drop the subject level
        epochs = epochs.loc[sub]
            
        gc.collect()
        
        return epochs
    
    print_("loading ... ", end='')
    alldata = pd.concat([load_subject(sub) for sub in subjects],
                        keys=subjects, names=['subject', 'trial', 'time'])
    print_('done.')
    
    if normalise == 'global':
        epochs_mean = pd.read_hdf(srcfile, 'epochs_mean')
        epochs_std = pd.read_hdf(srcfile, 'epochs_std')
        
        alldata = (alldata - epochs_mean[labels]) / epochs_std[labels]
        
    return alldata, epochtimes


def normalise_magnitudes(data, shift=True):
    """Normalise magnitudes to min=0 and mean=1 within columns of data."""
    
    signs = np.sign(data)
    magnitudes = np.abs(data)
    
    if shift:
        zero = np.min(magnitudes, axis=0)
    else:
        zero = np.zeros(magnitudes.shape[1])
        
    scale = np.mean(magnitudes - zero, axis=0)
    
    normed = (magnitudes - zero) / scale
    
    return normed * signs, zero, scale