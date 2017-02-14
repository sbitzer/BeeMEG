#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:43:48 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
import regressors
import seaborn as sns


#%% 
def get_trial_DM(dots=5, subjects=None, r_names=['dot_y', 'surprise', 
                 'logpost_left', 'entropy', 'trial_time', 'intercept']):
    if subjects is None:
        subjects = regressors.subjecti
    else:
        subjects = np.atleast_1d(subjects)
        
    dots = np.atleast_1d(dots)
    
    if r_names == 'all':
        r_names = np.concatenate((regressors.trial.columns, 
                                  regressors.trial_dot.columns,
                                  regressors.subject_trial.columns,
                                  regressors.subject_trial_dot.columns))

    # selected regressor names sorted by dependence
    tr_rn = np.intersect1d(r_names, regressors.trial.columns)
    tr_d_rn = np.intersect1d(r_names, regressors.trial_dot.columns)
    s_tr_rn = np.intersect1d(r_names, regressors.subject_trial.columns)
    s_tr_d_rn = np.intersect1d(r_names, regressors.subject_trial_dot.columns)

    DM = []
    for sub in subjects:
        dm = pd.concat([regressors.trial[tr_rn], 
                        regressors.subject_trial.loc[sub, s_tr_rn]], 
                       axis=1)
        for dot in dots:
            tr_d = regressors.trial_dot.xs(dot, level='dot')[tr_d_rn]
            tr_d.columns = tr_d.columns.map(lambda x: x+'_%d' % (dot, ))
            
            s_tr_d = regressors.subject_trial_dot.loc[sub, s_tr_d_rn].xs(dot, level='dot')
            s_tr_d.columns = s_tr_d.columns.map(lambda x: x+'_%d' % (dot, ))
            
            dm = pd.concat([dm, tr_d, s_tr_d], axis=1)
            
        dm['subject'] = sub
        dm.set_index('subject', append=True, inplace=True)
        dm = dm.reorder_levels(['subject', 'trial'])
        
        DM.append(dm)
        
    DM = pd.concat(DM)
    
    if 'intercept' in r_names:
        DM['intercept'] = 1
    
    return DM
            

def get_ithdot_DM_single_sub(sub, doti=5, intercept=True):
    # subject unspecific
    DM = pd.concat([regressors.trial_dot.xs(doti, level='dot'), 
                    regressors.trial,
                    regressors.subject_trial.loc[sub],
                    regressors.subject_trial_dot.loc[sub].xs(doti, level='dot')
                   ], axis=1)
    if intercept:
        DM['intercept'] = 1.0
    
    return DM


def get_ithdot_DM(doti=5, r_names=['dot_y', 'surprise', 'logpost_left', 
                                   'entropy', 'trial_time', 'intercept']):
    if 'intercept' in r_names:
        intercept = True
    else:
        intercept = False
    
    fresh = True
    for sub in regressors.subjecti:
        dm = get_ithdot_DM_single_sub(sub, doti, intercept)
        dm = dm[r_names]
        dm['subject'] = sub
        dm.set_index('subject', append=True, inplace=True)
        dm = dm.reorder_levels(['subject', 'trial'])
        
        if fresh:
            DM = dm
            fresh = False
        else:
            DM = pd.concat([DM, dm])
            
    return DM

# run as script, if called
if __name__ == '__main__':
    DM = get_ithdot_DM_single_sub(2)
    img = sns.plt.matshow(DM.corr().abs())