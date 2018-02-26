#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:25:44 2018

@author: bitzer
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% analysis function doing the actual work
def find_eog_events(epochs, channel, sacdur=120, perc=97, maxnpeak=3, 
                    exclude_edge=100):
    # pick horizontal EOG
    heog = mne.pick_channels(epochs.ch_names, [channel])
    
    eog = epochs.copy()
    
    # remove drift with high-pass filter, cutoff = 2 Hz
    eog.filter(2, None, fir_design='firwin', picks=heog)
    
    eog.pick_channels([channel])
    
    eog_amp = eog.copy()
    eog_amp._data = np.abs(eog_amp._data)
    
    h_freq = 1 / (sacdur * 2 / 1000)
    eog_amp.savgol_filter(h_freq)
    
    thresh = np.percentile(eog_amp._data, perc)
    ampdata = eog_amp._data.copy()
    ampdata[ampdata < thresh] = ampdata[ampdata < thresh]  / 100
    
    T = eog.times.size
    dt = (eog.times[1] - eog.times[0]) * 1000
    nsteps = int(np.round(sacdur / dt)) // 2
    
    if exclude_edge:
        # transform ms to time steps
        exclude_edge = int(exclude_edge // dt)
    
    # saccade events will be centered around the estimated peak
    sac_times = (np.arange(nsteps * 2) * dt - nsteps * dt).astype(int)
    
    eog_events = pd.DataFrame([], 
                index=pd.MultiIndex.from_arrays(
                        [[], []], names=['trial', 'time']),
                columns=pd.Index(sac_times, name='evtime'), dtype=float)
    
    # switch off info messages about no peaks found
    loglev = mne.utils.logger.level
    mne.utils.logger.setLevel('WARNING')
    for ep in range(eog._data.shape[0]):
        pis, pa = mne.preprocessing.peak_finder.peak_finder(
                ampdata[ep, 0, :], thresh)
        
        pis = np.array(pis)
        
        # ignore epochs in which there are too many peaks
        if pis.size > maxnpeak:
            continue
        elif pis.size > 0:
            for i, pi in enumerate(pis):
                # exclude peaks located right at the edges of the epoch, 
                # if desired
                if exclude_edge and (   pi <= exclude_edge 
                                     or pi >= T - 1 - exclude_edge):
                    continue
                
                # only add later events when the previous peak is at least 
                # sacdur away
                if i > 0 and (eog.times[pi] - eog.times[pis[i-1]]) < (
                        sacdur / 1000):
                    continue
                
                # figure out the available data time points centred around the 
                # event peak
                starti = max(pi - nsteps, 0)
                endi = min(pi + nsteps, T)
                sati = sac_times[  max(0, nsteps - pi) 
                                 : nsteps * 2 + min(0, T - nsteps - pi)]
                
                # ep + 1, because trial counts start at 1
                eog_events.loc[(ep + 1, int(eog.times[pi] * 1000)), sati] = (
                        eog._data[ep, 0, starti : endi])
    
    mne.utils.logger.setLevel(loglev)
        
    return eog_events, eog, eog_amp


#%% function for visual inspection of results
def show_events(eog, eog_events, trials, sacdur=120):
    trials = np.array(trials)
    fig, axes = plt.subplots(trials.size, 1, sharex=True, sharey=True,
                             squeeze=False, figsize=(7, 9.5))
    
    for trial, ax in zip(trials, axes[:, 0]):
        ax.plot(eog.times * 1000, eog._data[trial - 1, 0, :])
        ax.set_ylabel(trial)
        
        try:
            sac = eog_events.loc[trial]
        except KeyError:
            pass
        else:
            yl = ax.get_ylim()
            for time in sac.index:
                ax.plot(np.r_[0, sacdur] - sacdur / 2 + time, 
                        np.ones(2) * np.diff(yl) * 0.1 + yl[0], 'k')
    
    fig.tight_layout()
    
    return fig, axes


#%% only run the complete analysis when running as script
if __name__ == "__main__":
    import helpers
    import scipy.stats
    import os
    import regressors
    
    
    #%% options
    # use this channel to find EOG events
    # EOG061 for horizontal EOG, EOG062 for vertical EOG
    channel = 'EOG061'
    
    # expected typical duration of a saccade in ms
    # this will influence grade of smoothing (low-pass cut-off) of EOG signal
    # and determines the number of time points around an event that are returned
    sacdur = 120
    
    # peak finding threshold as percentile of smoothed EOG amplitudes
    perc = 97
    
    # ignore events in trials with more than this number of detected events
    # this is to guard against extremely noisy epochs
    maxnpeak = 3
    
    # time in ms that is excluded from event detection from either end of the epoch
    # this is to remove events clustering around the edges of the epoch time window
    # which are presumed to be filtering artefacts; set to 0 to switch off
    exclude_edge = 100
    
    # where to store
    file = pd.datetime.now().strftime('eog'+'_%Y%m%d%H%M'+'.h5')
    file = os.path.join(helpers.resultsdir, file)
    
    with pd.HDFStore(file, mode='w', complevel=7, complib='blosc') as store:
        store['scalar_params'] = pd.Series(
                [sacdur, perc, maxnpeak, exclude_edge],
                ['sacdur', 'perc', 'maxnpeak', 'exclude_edge'])
        store['channel'] = pd.Series(channel)
    
    
    #%% run the analysis
    subjects = helpers.find_available_subjects(megdatadir=helpers.megdatadir)
    measures = pd.DataFrame([], index=subjects,
                            columns=['r_times', 'r_times_p', 'r_signs', 
                                     'r_signs_p', 'N', 'Nuniq'])
    eog_events = []
    trial_times = []
    for sub in subjects:
        print('\nprocessing eog for subject %2d' % sub, flush=True)
        print('-----------------------------', flush=True)
        
        epochs = helpers.load_meg_epochs_from_sdat(sub)
        eog_ev, eog, eog_amp = find_eog_events(epochs, channel, sacdur, perc, 
                                                   maxnpeak, exclude_edge)
        eog_events.append(eog_ev)
        
        evtimes = eog_ev.index.get_level_values('time').values
        evtrials = eog_ev.index.get_level_values('trial').unique()
        
        # if there are several events in one trial, 
        # select the eog event that has the largest amplitude
        evtimes_unique = pd.Series(np.zeros(evtrials.size, dtype=int), 
                                   index=evtrials, name='event_time')
        evsigns_unique = pd.Series(np.zeros(evtrials.size, dtype=int), 
                                   index=evtrials, name='event_sign')
        for tr in evtrials:
            evtimes_unique.loc[tr] = eog_ev.loc[tr].abs().mean(axis=1).idxmax()
            evsigns_unique.loc[tr] = np.sign(eog_ev.loc[
                    (tr, evtimes_unique.loc[tr])].mean())
    
        rts = regressors.subject_trial.RT.loc[sub].loc[evtrials]
        chs = regressors.subject_trial.response.loc[sub].loc[evtrials]
        
        trial_times.append(pd.concat([evtimes_unique / 1000, evsigns_unique, rts, 
                                      chs], axis=1))
        
        measures.loc[sub, ['r_times', 'r_times_p']] = scipy.stats.pearsonr(
                evtimes_unique, rts)
        measures.loc[sub, ['r_signs', 'r_signs_p']] = scipy.stats.pearsonr(
                evsigns_unique, chs)
        measures.loc[sub, 'N'] = eog_ev.shape[0]
        measures.loc[sub, 'Nuniq'] = evtrials.size
        
    measures = measures.infer_objects()
    
    eog_events = pd.concat(eog_events, keys=subjects, 
                           names=['subject', 'trial', 'time'])
    trial_times = pd.concat(trial_times, keys=subjects, names=['subject', 'trial'])
    
    
    with pd.HDFStore(file, mode='r+', complevel=7, complib='blosc') as store:
        store['measures'] = measures
        store['trial_times'] = trial_times
        store['eog_events'] = eog_events
        
