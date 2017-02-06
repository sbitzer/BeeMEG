#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:43:48 2017

@author: bitzer
"""

import pandas as pd
import regressors
import seaborn as sns


#%% 
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