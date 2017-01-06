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
import rtmodels

sub = 2

respRT = helpers.load_subject_data(sub)

dotpos, dotstd, dotdt = helpers.load_dots(dotdir='data')
D = dotpos.shape[0]

#%% make batch indices: a batch collects all trials with common dot positions
#   except for the 5th dot position

# there were 28 'long trials' for which we modulated 5th dot positions
batches = np.arange(1, 29)
B = batches.size

# there were 6 different dot positions
batches = np.tile(batches, 6)

# after the 5th dot modulated trials there were 72 "short trials"
batches = np.r_[batches, np.arange(29, 29+72)]

# then these trials were repeated, but with opposite sign of the x-coordinate
# conveniently, the batch numbers can restart at 101
batches = np.r_[batches, 100+batches]


#%% get "correct" decision after 4th dot from ideal observer
cond = helpers.defaultcond
feature_means = np.c_[[-cond, 0], [cond, 0]]
model = rtmodels.discrete_static_gauss(dt=dotdt, maxrt=dotdt*D, 
                                       toresponse=helpers.defaultto, choices=[-1, 1], 
                                       Trials=dotpos, means=feature_means)

# set ideal observer parameters
model.intstd = dotstd
model.noisestd = 1e-15
model.prior = 0.5

# generate log-posterior beliefs
logpost, _ = model.compute_logpost_from_features(np.arange(480))

# get "correct" choice after 4th dot as the choice with the largest posterior
# belief at that time point; this is the same as summing the x-coordinates of
# the first 4 dots and checking whether the sum is positive: if yes, correct 
# choice is 1 else -1
correct_4th = model.choices[np.argmax(logpost[3, :, :, 0], axis=0)]

# make into data frame for easier handling of indeces
correct_4th = pd.DataFrame(correct_4th, index=pd.Index(np.arange(480)+1, 
                           name='trial'), columns=['correct_4th'])
respRT_ext = pd.concat([respRT, correct_4th], axis=1)


#%% define penalty that is 0 for an immediate correct response, maximal for an
#   immediate wrong response and is continuous in between

# for correct choices, the penalty is equal to the RT
ind = respRT_ext.response == respRT_ext.correct_4th
respRT_ext.loc[ind, 'penalty'] = respRT.loc[ind, 'RT']

# for wrong choices, the penalty is 2*maxrt - RT
ind = respRT_ext.response == -respRT_ext.correct_4th
respRT_ext.loc[ind, 'penalty'] = 2*model.maxrt - respRT.loc[ind, 'RT']

# timed-out responses have intermediate penalty
ind = respRT_ext.response == 0
respRT_ext.loc[ind, 'penalty'] = model.maxrt


#%% add 5th dot position and check correlations
respRT_ext['5th_dot'] = dotpos[4, 0, :]

# The penalty is defined in terms of the correct choice according to the first 
# 4 dot positions, so it makes sense to define the 5th dot position in relation
# to whether it supports the correct choice, or not. If the correct choice is
# right (1) a positive dot position (x>0) supports the correct choice and 
# should lead to a low penalty, but if the correct choice is left (-1), a 
# positive dot position supports the wrong choice and should lead to a high
# penalty. By multiplying the 5th dot position with the (sign of) the correct
# choice we can flip that relationship such that a larger value always 
# indicates greater support for the correct choice.
respRT_ext['support_correct'] = dotpos[4, 0, :] * respRT_ext['correct_4th']

sns.jointplot(x='support_correct', y='penalty', data=respRT_ext)


#%% bin 5th dot positions and compute mean penalty in bins
#   the manipulated 5th dot positions were 64 pixels apart and the used values
#   were [25, -25] + [-160, -96, -32, 32, 96, 160]
bins = np.r_[-300, np.arange(-160, 200, 64), 300]
B = bins.size-1
respRT_ext['support_correct_bin'] = np.digitize(respRT_ext['support_correct'], bins)

binpenalty = np.zeros(B)

for i in range(B):
    binpenalty[i] = respRT_ext.loc[respRT_ext['support_correct_bin']==i+1, 'penalty'].mean()
    
sns.plt.plot(binpenalty)
sns.plt.xlabel('support for correct choice')
sns.plt.ylabel('mean penalty')