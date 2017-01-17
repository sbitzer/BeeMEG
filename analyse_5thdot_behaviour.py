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

dotpos = helpers.load_dots(dotdir='data')
maxrt = dotpos.shape[0] * helpers.dotdt
trial_info = helpers.get_5th_dot_infos(dotpos)
randind = np.random.randint(1, 481, size=480)
#trial_info.support_correct = trial_info.loc[randind, 'support_correct'].values
#trial_info.support_correct_bin = trial_info.loc[randind, 'support_correct_bin'].values
                                            
# use "None" for all subjects; for a single subject (e.g. "[24]") plots will
# be shown
subjects = [14]

if subjects is None:
    subjects = helpers.find_available_subjects()
subjects = np.sort(np.array(subjects))
S = subjects.size
    
subject_info = pd.DataFrame([], index=pd.Index(subjects, name='subject'))

for si, sub in enumerate(subjects):
    respRT = helpers.load_subject_data(sub)
    respRT_ext = pd.concat([respRT, trial_info], axis=1)
    
    #%% check whether subject responded correctly according to first 4 dots
    respRT_ext['is_correct'] = (respRT_ext.correct_4th == respRT_ext.response).astype(int)
    
    
    #%% define penalty that is 0 for an immediate correct response, maximal for an
    #   immediate wrong response and is continuous in between
    
    # for correct choices, the penalty is equal to the RT
    ind = respRT_ext.response == respRT_ext.correct_4th
    respRT_ext.loc[ind, 'penalty'] = respRT_ext.loc[ind, 'RT']
    
    # for wrong choices, the penalty is 2*maxrt - RT
    ind = respRT_ext.response == -respRT_ext.correct_4th
    respRT_ext.loc[ind, 'penalty'] = 2*maxrt - respRT_ext.loc[ind, 'RT']
    
    # timed-out responses have intermediate penalty
    ind = respRT_ext.response == 0
    respRT_ext.loc[ind, 'penalty'] = maxrt
    
    
    #%% check correlation
    if S == 1:
        sns.jointplot(x='support_correct', y='penalty', data=respRT_ext)
    
    
    #%% compute mean penalty, fraction correct and median RT in 5th dot bins
    B = respRT_ext['support_correct_bin'].unique().size
    binpenalty = np.zeros(B)
    frac_correct = np.zeros(B)
    medianRT = np.zeros(B)
    
    for i in range(B):
        binind = respRT_ext['support_correct_bin']==i+1
        binpenalty[i] = respRT_ext.loc[binind, 'penalty'].mean()
        frac_correct[i] = np.mean(respRT_ext.loc[binind, 'correct_4th'] == 
                                  respRT_ext.loc[binind, 'response'])
        medianRT[i] = respRT_ext.loc[binind, 'RT'].median()
    
    if S == 1:
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
        sns.plt.xlabel('support for correct choice')
        sns.plt.ylabel('median RT')
    
    
    #%% compute logitstic regression
    logitres = logit('is_correct ~ support_correct', respRT_ext).fit()
    subject_info.loc[sub, 'logit_intercept'] = logitres.params.Intercept
    subject_info.loc[sub, 'logit_coef'] = logitres.params.support_correct
    subject_info.loc[sub, 'logit_p'] = logitres.llr_pvalue
    if S == 1:
        print(logitres.summary())
        
sns.plt.show()