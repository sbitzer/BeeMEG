#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:45:48 2017

@author: bitzer
"""

import os
import numpy as np
import pandas as pd
import helpers
import posterior_model_measures


#%%
subjecti = pd.Index(helpers.find_available_subjects(behavdatadir=
    helpers.behavdatadir), name='subject')
triali = pd.Index(np.arange(480) + 1, name='trial')
doti = pd.Index(np.arange(25)+1, name='dot')

dotpos = helpers.load_dots()
trial_info = helpers.get_5th_dot_infos(dotpos)

infresult_file = 'infres_collapse_201612121759.npz'


#%% trial

# correct
subinfo = helpers.load_subject_data(2)
trial = pd.DataFrame(subinfo['stimulus']).sort_index()
trial.rename(columns={'stimulus': 'correct'}, inplace=True)

#%% trial x dots
trial_dots = pd.DataFrame([], index=pd.MultiIndex.from_product([triali, doti], 
                          names=['trial', 'dot']))

# dotpos
trial_dots['dot_x'] = dotpos[:, 0, :].flatten('F')
trial_dots['dot_y'] = dotpos[:, 1, :].flatten('F')

# support_correct
trial_dots['support_correct'] = (dotpos[:, 0, :] * trial.correct.values).flatten('F')

# create ideal observer model
model = helpers.get_ideal_observer()
# generate log-posterior beliefs of ideal observer
logpost, _ = model.compute_logpost_from_features(np.arange(480))

# get "correct" choice after having seen dots as the choice with the largest posterior
# belief at that time point; this is the same as summing the x-coordinates of
# the first 4 dots and checking whether the sum is positive: if yes, correct 
# choice is 1 else -1; note that the logpost will probably never be exactly 0
# (noisestd is set to 1e-15) which is the case in rare cases for the cumulative
# sum of dot positions; therefore, the sign of the cumulative sum of dot 
# positions is sometimes (rarely) 0 while correct_ideal here is always 1 or -1
trial_dots['correct_ideal'] = model.choices[np.argmax(logpost[:, :, :, 0], 
    axis=1)].flatten('F')

# support_previous_ideal
trial_dots['support_previous_ideal'] = trial_dots.correct_ideal * trial_dots.dot_x


#%% subject

# accuracy, median RT, std RT
allresp = helpers.load_all_responses()
subject = pd.DataFrame({'accuracy': allresp['correct'].mean(level='subject'),
                        'medianRT': allresp['RT'].median(level='subject'),
                        'stdRT': allresp['RT'].std(level='subject')})

# accuracy for ideal observer correct
correct_ideal, _ = model.gen_response_from_logpost(logpost)
# necessary, because order of trials is still as in experiment
allresp_sorted = allresp.sort_index()
# 
subject['accuracy_ideal'] = (allresp_sorted.response.values.reshape((36, 480), 
    order='C') == correct_ideal[:, 0]).mean(1)

# model parameters
with np.load(os.path.join(helpers.resultsdir, infresult_file)) as data:
    ep_summary = pd.DataFrame(data['ep_summary'].astype(float), index=subjecti, 
                              columns=data['ep_summary_cols'].astype(str))
subject = pd.concat([subject, ep_summary[['ndtmode', 'dtmode', 'ndt_vs_RT', 
    'ndtspread', 'noisestd', 'bound', 'bshape', 'bstretch', 'prior', 
    'lapseprob', 'lapsetoprob']]], axis=1)


#%% subject x trial

# response, RT and is_correct
subject_trial = allresp_sorted.copy()
subject_trial.rename(columns={'correct': 'is_correct'}, inplace=True)

# is_correct_ideal
subject_trial['is_correct_ideal'] = (allresp_sorted.response.values.reshape(
    (36, 480), order='C') == correct_ideal[:, 0]).flatten(order='C')

# trial index (time in experiment)
for sub in subjecti:
    # +1 so that time in trials starts at 1
    subject_trial.loc[sub, 'trial_time'] = allresp.loc[sub, :].index.argsort() + 1

subject_trial['trial_time'] = subject_trial['trial_time'].astype('int32')

# posterior entropy
subject_trial['entropy'] = posterior_model_measures.get_entropies().flatten(order='C')


#%% subject x trial x dots

subject_trial_dot = posterior_model_measures.get_dot_level_measures()
