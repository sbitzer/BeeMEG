#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:45:48 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import helpers
import posterior_entropies


#%%
subjecti = pd.Index(helpers.find_available_subjects(behavdatadir=
    helpers.behavdatadir), name='subject')
triali = pd.Index(np.arange(480) + 1, name='trial')
doti = pd.Index(np.arange(25)+1, name='dot')

dotpos = helpers.load_dots()
trial_info = helpers.get_5th_dot_infos(dotpos)


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
# model parameters

# median RT

# std RT

# accuracy


#%% subject x trial
# posterior entropy

# response

# error

# RT

# trial index (time in experiment)


#%% subject x trial x dots
# surprise

# log_posterior_odds

# momentary_evidence
