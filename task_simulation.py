#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:57:11 2017

@author: bitzer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%% set screen dpi
# you can use https://www.infobyip.com/detectmonitordpi.php to determine 
# approximate dpi for your screen

# my desktop screen
dpi = 92
dotsize = 10

# beamer


#%% define stimulus and units
# distance of targets from centre of screen
v_target = 0.6 / 180 * np.pi
# standard deviation of dot movement in x and y
v_std = 1.7 / 180 * np.pi
# size of figure window
v_window = 15 / 180 * np.pi

screendist = 90 # cm

v2cm = lambda v: np.tan(v) * screendist
v2in = lambda v: v2cm(v) / 2.54


#%% setup figure window and axis
fig = plt.figure(figsize=(v2in(v_window), v2in(v_window)), dpi=92)
ax =  plt.axes([0, 0, 1, 1], facecolor='k')
ax.set_aspect('equal', 'datalim')

# this sets axis units to cm and makes centre at (0, 0)
ax.set_xlim(-v2cm(v_window) / 2, v2cm(v_window) / 2)
ax.set_ylim(-v2cm(v_window) / 2, v2cm(v_window) / 2)


#%% pre-plot all presentation elements
targets, = ax.plot([-v2cm(v_target), v2cm(v_target)], [0, 0], '.', 
                  mfc='#FFD42A', mew=0, ms=dotsize)
dot, = ax.plot(1, 1, '.', mfc='w', mew=0, ms=dotsize)
fix, = ax.plot(0, 0, '+', mec='w', mew=dotsize/6, ms=dotsize)


#%% define the animation class 
class Trial(object):
    # all durations and times given in 100 ms, i.e. 7 = 700 ms
    
    def __init__(self, targets, fix, dot):
        # handles to plot components
        self.targets = targets
        self.fix = fix
        self.dot = dot
        
        self.targets.set_visible(False)
        self.dot.set_visible(False)
        self.fix.set_visible(False)
        
        self.targetfix_dur = 7
        
        # trial count
        self.count = 0
    
    def start(self):
        self.count += 1
        
        self.targets.set_visible(False)
        self.fix.set_visible(True)
        self.dot.set_visible(False)
        
        # sample timings
        self.onlyfix_dur = np.random.randint(12, 16)
        self.dot_onset = self.onlyfix_dur + self.targetfix_dur + 1
        
        # sample correct target for this trial
        self.correct = np.sign(np.random.rand()-0.5)
        print('trial %2d correct = % d' % (self.count, self.correct))
        
        return [self.targets, self.fix, self.dot]
    
        
    def draw_frame(self, time):
        if time > self.onlyfix_dur:
            if time == self.onlyfix_dur + 1:
                self.targets.set_visible(True)
            if time >= self.dot_onset:
                if time == self.dot_onset:
                    self.fix.set_visible(False)
                    self.dot.set_visible(True)
                elif time < self.dot_onset + 25:
                    # sample dot position
                    dotpos = np.random.randn(2) * v2cm(v_std)
                    dotpos[0] += self.correct * v2cm(v_target)
                    
                    # set dot position
                    self.dot.set_data(dotpos)
                elif time == self.dot_onset + 25:
                    self.targets.set_visible(False)
                    self.dot.set_visible(False)
                    
        return [self.targets, self.fix, self.dot]
                

#%% run animation
trial = Trial(targets, fix, dot)
anim = FuncAnimation(fig, trial.draw_frame, init_func=trial.start,
                     frames=np.arange(1, 48), interval=100, blit=True)