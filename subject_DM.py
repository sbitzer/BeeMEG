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
import helpers


#%% 
def get_trial_time_DM(r_names, trt, subjects=None, delay=0, 
                      unobserved_val=np.nan, normalise=False, exclude_to=False,
                      resp_align=False):
    if np.isscalar(delay):
        delay = {r_name: delay for r_name in r_names}
    
    assert set(r_names) == delay.keys()
    
    if resp_align:
        exclude_to = True
    
    # collect all regressors
    DM = []
    add_intercept = False
    for r_name in r_names:
        if r_name == 'intercept':
            add_intercept = True
        elif r_name.endswith('_time'):
            DM.append(regressors.dot_to_time(trt, delay[r_name], r_name, 
                                             unobserved_val))
        else:
            DM.append(regressors.subject_trial_time[r_name](trt))
        
    # concatenate them into DataFrame
    DM = pd.concat(DM, axis=1)
    
    if subjects is not None and len(subjects) > 0:
        DM = DM.loc[subjects]
        
    if add_intercept:
        DM['intercept'] = 1
    
    if exclude_to or resp_align:
        rts = regressors.subject_trial.RT.loc[subjects]
        
    if exclude_to:
        # exclude trials in which there was no response (timed out trials)
        def get_response_trials(sub):
            DMsub = DM.loc[sub]
            rtssub = rts.loc[sub]
            return DMsub.loc[
                    (rtssub[rtssub != helpers.toresponse[1]].index, 
                     slice(None)), :]
        
        DM = pd.concat([get_response_trials(sub) for sub in subjects],
                        keys=subjects, names=['subject', 'trial', 'time'])
        
    if resp_align:
        DM.index = helpers.to_resp_aligned_index(DM.index, rts, in_units='s')
        
    return normalise_DM(DM, normalise)


def get_trial_DM(dots=5, subjects=None, r_names=['dot_y', 'surprise', 
                 'logpost_left', 'entropy', 'trial_time', 'intercept'],
                 normalise=False):
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
    
    return normalise_DM(DM, normalise)
            

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


#%% normalisation
def normfun(regressor, msratio=2):
    mean = regressor.mean()
    std = regressor.std()
    
    # don't normalise constant regressors (the intercept)
    if std==0:
        return regressor
    else:
        # subtract mean, if you can get into trouble with collinearity with the 
        # intercept, otherwise leave it be
        if abs(mean) > msratio * std:
            # subtracting the mean ensures that the regressor is orthogonal to the
            # intercept, any average effect of the regressor will move to the 
            # intercept such that only the variance of the signal with the 
            # regressor remains
            regressor = regressor - mean
        
        # scale regressor so that it's standard deviation = 1
        return regressor / std
    

def normalise_DM(DM, normalise, msratio=2):
    if normalise:
        # ensure that DM is a DataFrame, because apply in Series only works on
        # single elements
        DM = pd.DataFrame(DM)
        
        if normalise == 'subject':
            subjects = DM.index.get_level_values('subject').unique()
            
            # if subject is not the outermost level, make it to that
            levelnames = DM.index.names
            if levelnames[0] != 'subject':
                DM = DM.reorder_levels(
                        ['subject'] + list(np.setdiff1d(levelnames, ['subject'])))
            
            # normalise within subject
            DM = pd.concat(
                    [DM.loc[sub].apply(normfun, args=(msratio,)) 
                     for sub in subjects],
                    keys=subjects,
                    names=DM.index.names)
            
            # if subject wasn't outermost level, restore old order
            if levelnames[0] != 'subject':
                DM = DM.reorder_levels(levelnames).sort_index()
                
            return DM
        else:
            return DM.apply(normfun, args=(msratio,))
    else:
        return DM


#%% collinearity
def cnumfun(DM):
    # ensure that array is standard numpy array
    DM = np.array(DM)
    
    # if there are fewer cases than there are regressors, the design matrix
    # cannot be full rank so that the condition number is infinity
    if DM.shape[0] < DM.shape[1]:
        return np.inf
    else:
        # only include columns in which at least one value is different from 0
        DM = DM[:, np.any(DM != 0, axis=0)]
        
        # ensure that regressor vectors have length 1
        DM = DM / np.sqrt((DM ** 2).sum(axis=0))
        
        eig = np.linalg.eigvals(np.dot(DM.T, DM))
        
        return np.sqrt(eig.max() / max(0, eig.min()))


def compute_condition_number(DM, scope='full'):
    if scope == 'full':
        return cnumfun(DM)
    elif scope == 'subject':
        subjects = DM.index.get_level_values('subject').unique()
        return pd.Series([cnumfun(DM.loc[sub]) for sub in subjects],
                         index=subjects, name='condition number')
    else:
        raise ValueError('The chosen scope option is not defined!')


#%%
# run as script, if called
if __name__ == '__main__':
    DM = get_ithdot_DM_single_sub(2)
    img = sns.plt.matshow(DM.corr().abs())