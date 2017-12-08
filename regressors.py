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

# summed dotpos, is proportional to summed log-likelihood ratio for dot_x, 
# for both x and y it's also a measure of the mean position within the trial
trial_dot['sum_dot_x'] = dotpos[:, 0, :].cumsum(axis=0).flatten('F')
trial_dot['sum_dot_y'] = dotpos[:, 1, :].cumsum(axis=0).flatten('F')

# summed dotpos up to the previous dot
trial_dot['sum_dot_x_prev'] = np.zeros(trial_dot.shape[0])
trial_dot.loc[(slice(None), slice(2, 25)), 'sum_dot_x_prev'] = \
    trial_dot.loc[(slice(None), slice(1, 24)), 'sum_dot_x'].values
trial_dot['sum_dot_y_prev'] = np.zeros(trial_dot.shape[0])
trial_dot.loc[(slice(None), slice(2, 25)), 'sum_dot_y_prev'] = \
    trial_dot.loc[(slice(None), slice(1, 24)), 'sum_dot_y'].values

# distance to center of screen
trial_dot['dot_dist'] = np.sqrt(trial_dot.dot_x ** 2 + trial_dot.dot_y ** 2)

# relative dot movement: taking previous dot as centre, where did the dot jump?
trial_dot['move_x'] = np.r_[np.zeros((1, dotpos.shape[2])), 
                            np.diff(dotpos[:, 0, :], axis=0)].flatten('F')
trial_dot['move_y'] = np.r_[np.zeros((1, dotpos.shape[2])), 
                            np.diff(dotpos[:, 1, :], axis=0)].flatten('F')

# hypothetical saccades to presented dot positions
trial_dot['saccade_x'] = trial_dot['move_x']
trial_dot.loc[(slice(None), 1), 'saccade_x'] = trial_dot.loc[(slice(None), 1), 
                                                             'dot_x']
trial_dot['saccade_y'] = trial_dot['move_y']
trial_dot.loc[(slice(None), 1), 'saccade_y'] = trial_dot.loc[(slice(None), 1), 
                                                             'dot_y']


# how far did the dot jump?
trial_dot['move_dist'] = np.r_[np.zeros((1, dotpos.shape[2])), 
                               np.sqrt(np.sum(np.diff(dotpos, axis=0) ** 2, axis=1))].flatten('F')

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

# accumulated evidence is the log posterior ratio from the previous dot
for sub in subjecti:
    subject_trial_dot.loc[(sub, slice(None), 1), 'accev'] = (
            np.log(subject.prior[sub]) - np.log(1 - subject.prior[sub]) )
subject_trial_dot.loc[(slice(None), slice(None), slice(2, None)), 'accev'] = \
        subject_trial_dot.loc[(slice(None), slice(None), slice(1, 24)), 'lpr'].values

# accumulated evidence Gram-Schmidt orthogonalised with respect to the 
# x-position of the corresponding dot
subject_trial_dot['lpr_orth'] = np.nan
for sub in subjecti:
    for dot in doti:
        Q, _ = np.linalg.qr(
                np.c_[trial_dot.loc[(slice(None), dot), 'dot_x'],
                      subject_trial_dot.loc[(sub, slice(None), dot), 'lpr']])
        
        # Q is the negative of the Gram-Schmidt orthonormalised vectors, see
        # https://cran.r-project.org/web/packages/matlib/vignettes/gramreg.html
        subject_trial_dot.loc[(sub, slice(None), dot), 'lpr_orth'] = -Q[:, 1]
del Q

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

# dot_x with sign flipped by the choice of the subject
subject_trial_dot['dot_x_cflip'] = (
        np.tile(trial_dot.dot_x.values, 36)
        * np.tile(subject_trial.response.values[:, None], [1, 25]).flatten())

# accev with sign flipped by the choice of the subject
subject_trial_dot['accev_cflip'] = (
        subject_trial_dot.accev.values
        * np.tile(subject_trial.response.values[:, None], [1, 25]).flatten())


#%% subject x trial x time
mprepsig = lambda t: np.fmax(  np.fmin(t + 0.6, 0.3) 
                             + np.fmin(np.fmax((t + 0.3) / 3, 0), 0.1) 
                             - np.fmax(4*t/3, 0), 0) / 0.4

def motoprep(trt):
    trt = np.atleast_1d(trt)
    
    mprep = pd.concat([
            pd.Series(mprepsig(t - subject_trial.RT.values), 
                      index=subject_trial.index)
            for t in trt],
            keys=trt, names=['time']+subject_trial.index.names)
    
    return mprep.reorder_levels(['subject', 'trial', 'time']).sort_index()


def motoresp(trt):
    trt = np.atleast_1d(trt)
    
    mprep = pd.concat([
            pd.Series(  mprepsig(t - subject_trial.RT.values)
                      * subject_trial.response.values, 
                      index=subject_trial.index)
            for t in trt],
            keys=trt, names=['time']+subject_trial.index.names)
    
    return mprep.reorder_levels(['subject', 'trial', 'time']).sort_index()

# number of dots
D = doti.max()
maxrt = helpers.dotdt * D
linprepsig = lambda x: x / maxrt + 1

def motoresp_lin(trt):
    """A signal that simply linearly increases throughout the trial, if the 
       response was right, and decreases, if the response was left, with a 
       fixed positive or negative height of 1 at response time and slope 
       chosen so that 0 is only reached for the largest allowed response time.
    """
    trt = np.atleast_1d(trt)
    
    mprep = pd.concat([
            pd.Series(  linprepsig(t - subject_trial.RT.values)
                      * subject_trial.response.values, 
                      index=subject_trial.index)
            for t in trt],
            keys=trt, names=['time']+subject_trial.index.names)
    
    return mprep.reorder_levels(['subject', 'trial', 'time']).sort_index()
       

dcsig = lambda t: np.fmax(np.floor(t * 10) + 1, 0)

def dotcount(trt):
    trt = np.atleast_1d(trt)
    
    dcs = dcsig(trt)
    
    dcs = pd.concat([
            pd.Series(np.fmin(dcsig(subject_trial.RT.values), dc), 
                      index=subject_trial.index) 
            for dc in dcs],
            keys=trt, names=['time']+subject_trial.index.names)
    
    return dcs.reorder_levels(['subject', 'trial', 'time']).sort_index()


def sumx_time(trt, delay=0.3, unobserved_val=np.nan):
    """Computes sum of x-coordinates at given time points.
    
        The assumption is that this represents accumulated evidence. The sum is
        updated whenever a new dot appeared on the screen, but a delay can be
        used to shift these updates to later time points. The delay will be the
        same for each dot so that the sum is still updated every dotdt 
        (100 ms).
        
        Time points which fall into [0, delay) will be set to unobserved_val.
        Also, time points starting with RT+delay will be set to this value to 
        account for the fact that no more dots were shown after the response.
    """
    reg = 'sum_dot_x'
    
    trt = np.atleast_1d(trt)
    
    # this associates each time point with a dot
    # discrete time points are assumed to belong to forward bins: 
    # time index of t stands for the time window [t, t+dt]
    # for delay=0, times 0-0.09 map to dot1, 0.1-0.19 to dot2 and so on
    # the delay shifts the series by the corresponding time steps
    
    # this is the precision of the time steps in decimals
    prec = int(np.ceil(-np.log10(np.diff(trt).min())))
    # how often is dotdt in the requested time? convert to interges first,
    # because otherwise numerical inaccuracies can shift the borders randomly
    toint = lambda x: np.array(x * 10**prec, dtype=int)
    dotinds = (toint(trt) - toint(delay)) // toint(helpers.dotdt)
    # map all values <=0 to dot1 and set dot25 as maximum
    dotinds = np.fmin(np.ceil(np.fmax(dotinds, 0)).astype(int) + 1, D)
    
    # get the sum_dot_x values
    accev = pd.concat([
            trial_dot[reg].loc[(slice(None), ind)] 
            for ind in dotinds],
            keys=trt, names=['time']+trial_dot.index.names[:1])
    
    # set all values at times < delay to unobserved_val
    accev.loc[(trt[trt < delay], slice(None))] = unobserved_val
    
    # bring to standard format and expand to every subject
    accev = accev.reorder_levels(['trial', 'time']).sort_index()
    accev = pd.concat([accev]*subjecti.size, 
                      keys=subjecti, 
                      names=['subject', 'trial', 'time'])
    
    # after RT no new dots were shown - assume that the dots shown up to 
    # that time point are processed with the selected delay, but then set 
    # to nan
    
    # get tend
    tend = subject_trial.RT + delay
    # loop over trials with RT so early that some requested
    # time points fall after RT+delay
    for sub, trial in tend[tend < trt.max()].index:
        # replace all values in accev with trt > tend with tend_val
        aloc = accev.loc[(sub, trial)].values
        aloc[trt > tend.loc[(sub, trial)]] = unobserved_val
        accev.loc[(sub, trial)] = aloc
        
    return accev
    

def dotx_time(trt, delay=0.3):
    """Computes assumed value of accumulated evidence at given time points.
    
        The assumption is that accumulated evidence will change after a given
        delay from the occurrence of a dot. I use sum_dot_x_prev as base 
        measure of accumulated evidence, because it fluctuates together with 
        the log posterior ratio (lpr) and only differs by the offset given by
        the bias which is constant for each subject and therefore anyway will
        only move into the intercept. Furthermore, the lpr is sometimes 
        undefined for late time points, because the model predicted that 
        responses will be made before these data points. This is no issue with
        sum_dot_x_prev.
        
        Will take a few seconds to compute, because I have to loop through 
        subjects and trials to limit the value of accumulated evidence for time
        points after the response.
    """
    
    trt = np.atleast_1d(trt)
    
    # only times from delay onwards need to be chosen from dot_x
    tind = trt > delay
    N0 = np.logical_not(tind).sum()
    
    dotinds = np.fmin(np.ceil(np.fmax(trt[tind] - delay, 0) / 0.1), D)
    
    dotx = pd.concat(
            [pd.Series(np.zeros(triali.size), index=triali) for i in range(N0)] 
            + [trial_dot.dot_x.loc[(slice(None), ind)] for ind in dotinds],
            keys=trt, names=['time', 'trial'])
    dotx = dotx.reorder_levels(['trial', 'time']).sort_index()
    dotx = pd.concat([dotx]*subjecti.size, 
                     keys=subjecti, 
                     names=['subject', 'trial', 'time'])
    
    # if trt is past RT+delay, dotx to 0
    
    # get tend
    tend = subject_trial.RT + delay
    
    for sub, trial in tend[tend < trt.max()].index:
        # replace all values in dotx with trt > tend with 0
        aloc = dotx.loc[(sub, trial)].values
        aloc[trt > tend.loc[(sub, trial)]] = 0
        dotx.loc[(sub, trial)] = aloc
            
    return dotx


subject_trial_time = {'motoprep': motoprep, 
                      'motoresponse': motoresp, 
                      'dotcount': dotcount,
                      'sumx_time': sumx_time,
                      'dotx_time': dotx_time}


#%% categorising regressors according to their name
def get_regressor_category(r_name):
    if r_name in trial.columns or r_name in subject_trial.columns:
        return 'trialreg'
    elif r_name in trial_dot.columns or r_name in subject_trial_dot.columns:
        return 'dotreg'
    elif r_name in ['motoprep']:
        return 'timereg'
    elif r_name in ['intercept', 'constant']:
        return 'constreg'

def get_regressor_categories(r_names):
    cats = ['trialreg', 'dotreg', 'timereg', 'constreg']
    
    return {cat: [r_name for r_name in r_names 
                  if cat==get_regressor_category(r_name)] 
            for cat in cats}
