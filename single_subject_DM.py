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
def get_ithdot_DM(sub, doti=5, intercept=True):
    # subject unspecific
    DM = pd.concat([regressors.trial_dot.xs(doti, level='dot'), 
                    regressors.trial,
                    regressors.subject_trial.loc[sub],
                    regressors.subject_trial_dot.loc[sub].xs(doti, level='dot')
                   ], axis=1)
    if intercept:
        DM['intercept'] = 1.0
    
    return DM


# run as script, if called
if __name__ == '__main__':
    DM = get_ithdot_DM(2)
    img = sns.plt.matshow(DM.corr().abs())