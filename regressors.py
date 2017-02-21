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
import scipy.stats


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
trial_dot = pd.DataFrame([], index=pd.MultiIndex.from_product([triali, doti], 
                          names=['trial', 'dot']))

# dotpos
trial_dot['dot_x'] = dotpos[:, 0, :].flatten('F')
trial_dot['dot_y'] = dotpos[:, 1, :].flatten('F')

# absolute dot-positions (measure distance to centre of axis; this is
# strongly related to momentary surprise about the corresponding x or y 
# component, the difference being that surprise is a parabola whereas the 
# absolute value is linear)
trial_dot['abs_dot_x'] = trial_dot.dot_x.abs()
trial_dot['abs_dot_y'] = trial_dot.dot_y.abs()

# support_correct
trial_dot['support_correct'] = (dotpos[:, 0, :] * trial.correct.values).flatten('F')

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
trial_dot['correct_ideal'] = model.choices[np.argmax(logpost[:, :, :, 0], 
    axis=1)].flatten('F')

# support_previous_ideal
trial_dot['support_previous_ideal'] = trial_dot.correct_ideal * trial_dot.dot_x

# computes the 2D-Gaussian log-likelihood with std given by model.intstd
gll = lambda m: scipy.stats.multivariate_normal.logpdf(
        np.c_[trial_dot.dot_x, trial_dot.dot_y], mean=m, 
        cov=np.eye(2) * model.intstd**2)

# momentary surprise about dot assuming ideal generative models
trial_dot['momentary_surprise'] = ( -np.log(0.5) - 
         np.logaddexp(gll(model.means[:, 0]), gll(model.means[:, 1])) )

# computes a 1D-Gaussian log-likelihood with std given by model.intstd
gll = lambda x, m: ( -np.log(np.sqrt(2*np.pi) * model.intstd) - 
                      0.5 * (x - m)**2 / model.intstd**2 )

# momentary surprise about y-position of dot assuming ideal generative models
trial_dot['momentary_surprise_y'] = -gll(trial_dot.dot_y, 0)

# momentary surprise about x-position according to ideal generative models
# ignoring uncertainty removed or induced by previous dots; the functional 
# form of this as a function of x-positions is very close to the corresponding
# form for y-positions: the form for x-positions is a bit shallower than that 
# for y-positions, but they are both parabolas centred at 0 with almost equal 
# range for common input
trial_dot['momentary_surprise_x'] = ( -np.log(0.5) - 
         np.logaddexp(gll(trial_dot.dot_x, model.means[0, 0]),
                      gll(trial_dot.dot_x, model.means[0, 1])))

# do PCA of momentary surprises; there are only two significant principal 
# components, the first one rests mostly on momentary_surprise with almost 
# equal contributions from the axis-specific surprises and the second one 
# represents the difference between momentary_surprise_x and momentary_surprise_y
# to check this, look at the columns of Vt.T, the variance explained by the 
# components is s / np.sqrt(sur.shape[0])
sur = trial_dot[['momentary_surprise', 'momentary_surprise_x', 'momentary_surprise_y']]
sur = sur - sur.mean()
U, s_ms, Vt_ms = np.linalg.svd(sur, full_matrices=False)
trial_dot['momentary_surprise_pc1'] = U[:, 0] * s_ms[0]
trial_dot['momentary_surprise_pc2'] = U[:, 1] * s_ms[1]

del U, sur, gll


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

# log-posterior ratio (corresponding to accumulated evidence in DDM)
subject_trial_dot['lpr'] = ( subject_trial_dot.logpost_left - 
                             subject_trial_dot.logpost_right )

# do PCA to find a regressor that is not strongly correlated with dot_x, but 
# still contains information about accumulated evidence
def pca(sub):
    df = pd.DataFrame(subject_trial_dot.loc[sub, 'lpr'])
    
    # use negative dot_x, because lpr is positive when left is probable
    # then lpr and negative dot_x will be correlated
    df['neg_dot_x'] = -trial_dot['dot_x'].values
    
    # only select the first 5 dots for the PCA, because they are most interesting
    # for the analysis; not doing this will mean that the accumulated evidence 
    # regressor will get too much weight (s[1] will be higher), because the 
    # accumulated evidence influences lpr stronger for the late dots while the 
    # first couple of dots are more dominated by the momentary evidence of dot_x,
    # this will make the two principal component regressors uncorrelated across all
    # dots but strongly correlated for the first 5 dots which also leads to a 
    # strong anti-correlation between dot_x and the accumulated evidence regressor
    df = df.loc[(slice(None), slice(1, 5)), :]

    # need to drop all nans for SVD
    df = df.dropna()
    
    # bring dot_x and lpr into comparable (normalised) spaces
    df = (df - df.mean()) / df.std()
    
    # perform SVD (Vt.T will be a rotation matrix implementing a 
    # counterclockwise rotation by exactly 3/4 pi, because the columns of Vt.T
    # have to be orthonormal)
    U, s, Vt = np.linalg.svd(df, full_matrices=False)
    
    return pd.Series(U[:, 1] * s[1], index=df.index, name='accev_pca')

# pd.concat will do the heavy lifting of integrating the indices
subject_trial_dot = pd.concat([subject_trial_dot, 
        pd.concat([pca(sub) for sub in subjecti], keys=subjecti)], axis=1)

# same approach for 'accumulated surprise' - the surprise that can be 
# attributed to a deviation from the acquired belief about the true target from
# the previous dots
def pca(sub):
    df = pd.DataFrame(subject_trial_dot.loc[sub, 'surprise'])
    
    df['momentary_surprise'] = trial_dot['momentary_surprise'].values
    
    # only select the first 5 dots for the PCA
    df = df.loc[(slice(None), slice(1, 5)), :]

    # need to drop all nans for SVD
    df = df.dropna()
    
    # remove means for PCA to work (dividing or not dividing by std only 
    # changes the scaling of the resulting regressor slightly)
    df = (df - df.mean()) / df.std()
    
    # perform SVD (Vt.T will be a rotation matrix implementing a 
    # counterclockwise rotation by exactly 3/4 pi, because the columns of Vt.T
    # have to be orthonormal)
    U, s, Vt = np.linalg.svd(df, full_matrices=False)
    
    return pd.Series(U[:, 1] * s[1], index=df.index, name='accsur_pca')

# pd.concat will do the heavy lifting of integrating the indices
subject_trial_dot = pd.concat([subject_trial_dot, 
        pd.concat([pca(sub) for sub in subjecti], keys=subjecti)], axis=1)
    
del pca