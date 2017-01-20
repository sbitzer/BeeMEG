#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:00:00 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import helpers
import seaborn as sns
from statsmodels.formula.api import logit
import sys

dotpos = helpers.load_dots(dotdir='data')
maxrt = dotpos.shape[0] * helpers.dotdt
trial_info = helpers.get_5th_dot_infos(dotpos)
randind = np.random.randint(1, 481, size=480)
#trial_info.support_correct = trial_info.loc[randind, 'support_correct'].values
#trial_info.support_correct_bin = trial_info.loc[randind, 'support_correct_bin'].values

# use "None" for all subjects; for a single subject (e.g. "[24]") plots will
# be shown
subjects = None
if len(sys.argv) > 1:
    if sys.argv[1] == 'None':
        subjects = None
    elif sys.argv[1][0] == '[':
        subjects = list(map(int, sys.argv[1][1:-1].split(',')))
    else:
        subjects = [int(sys.argv[1])]
    
if subjects is None:
    subjects = helpers.find_available_subjects(helpers.defaultdir)
subjects = np.sort(np.atleast_1d(subjects))
S = subjects.size
    
subject_info = pd.DataFrame([], index=pd.Index(subjects, name='subject'))

for si, sub in enumerate(subjects):
    respRT = helpers.load_subject_data(sub)
    respRT_ext = pd.concat([respRT, trial_info], axis=1)
    
    #%% drop trials in which the subject responded too fast
    respRT_ext.drop(respRT_ext.index[respRT_ext.RT < 0.55], inplace=True)
    
    
    #%% check whether subject responded correctly according to first 4 dots
    respRT_ext['is_correct_4th'] = (respRT_ext.correct_4th == 
        respRT_ext.response).astype(int)
    respRT_ext['is_correct'] = (respRT_ext.stimulus == 
        respRT_ext.response).astype(int)
    
    
    #%% define penalty that is 0 for an immediate correct response, maximal for an
    #   immediate wrong response and is continuous in between
    
    # for correct choices, the penalty is equal to the RT
    ind = respRT_ext.response == respRT_ext.correct_4th
    respRT_ext.loc[ind, 'penalty_4th'] = respRT_ext.loc[ind, 'RT']
    ind = respRT_ext.response == respRT_ext.stimulus
    respRT_ext.loc[ind, 'penalty'] = respRT_ext.loc[ind, 'RT']
    
    # for wrong choices, the penalty is 2*maxrt - RT
    ind = respRT_ext.response == -respRT_ext.correct_4th
    respRT_ext.loc[ind, 'penalty_4th'] = 2*maxrt - respRT_ext.loc[ind, 'RT']
    ind = respRT_ext.response == -respRT_ext.stimulus
    respRT_ext.loc[ind, 'penalty'] = 2*maxrt - respRT_ext.loc[ind, 'RT']
    
    # timed-out responses have intermediate penalty
    ind = respRT_ext.response == 0
    respRT_ext.loc[ind, 'penalty_4th'] = maxrt
    respRT_ext.loc[ind, 'penalty'] = maxrt
    
    
    #%% check correlation
    if S == 1:
        sns.jointplot(x='support_correct_4th', y='penalty_4th', data=respRT_ext)
    
    
        #%% compute mean penalty, fraction correct and median RT in 5th dot bins
        B = respRT_ext['support_correct_bin_4th'].unique().size
        binpenalty = np.zeros(B)
        frac_correct = np.zeros(B)
        medianRT = np.zeros((B, 3))
        median_names = ['full bin', 'follow 5th dot', 'against 5th dot']
        
        for i in range(B):
            binind = respRT_ext['support_correct_bin_4th']==i+1
            binpenalty[i] = respRT_ext.loc[binind, 'penalty_4th'].mean()
            frac_correct[i] = np.mean(respRT_ext.loc[binind, 'correct_4th'] == 
                                      respRT_ext.loc[binind, 'response'])
            medianRT[i, 0] = respRT_ext.loc[binind, 'RT'].median()
            medianRT[i, 1] = respRT_ext.loc[binind & (respRT_ext.response ==
                np.sign(respRT_ext['5th_dot'])), 'RT'].median()
            medianRT[i, 2] = respRT_ext.loc[binind & (respRT_ext.response !=
                np.sign(respRT_ext['5th_dot'])), 'RT'].median()
    
        sns.plt.figure()
        sns.plt.plot(binpenalty)
        sns.plt.xlabel('support for correct choice')
        sns.plt.ylabel('mean penalty')
        
        sns.plt.figure()
        sns.plt.plot(frac_correct)
        sns.plt.xlabel('support for correct choice')
        sns.plt.ylabel('fraction correct')
        
        sns.plt.figure()
        sns.plt.plot(medianRT)
        sns.plt.legend(median_names)
        sns.plt.xlabel('support for correct choice')
        sns.plt.ylabel('median RT')
    
    
    #%% compute logitstic regression
    logitres_4th = logit('is_correct_4th ~ support_correct_4th', respRT_ext).fit()
    subject_info.loc[sub, 'logit_intercept_4th'] = logitres_4th.params.Intercept
    subject_info.loc[sub, 'logit_coef_4th'] = logitres_4th.params.support_correct_4th
    subject_info.loc[sub, 'logit_p_4th'] = logitres_4th.llr_pvalue
    logitres = logit('is_correct ~ support_correct', respRT_ext).fit()
    subject_info.loc[sub, 'logit_intercept'] = logitres.params.Intercept
    subject_info.loc[sub, 'logit_coef'] = logitres.params.support_correct
    subject_info.loc[sub, 'logit_p'] = logitres.llr_pvalue
    if S == 1:
        print(logitres_4th.summary())
        
    
        
sns.plt.show()