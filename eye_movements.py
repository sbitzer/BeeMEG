#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:45:05 2017

@author: bitzer
"""

import numpy as np
import pandas as pd
from collections import deque
import regressors as reg


#%%
"""
according to Dimigen2009 microsaccades may execute in less than 50 ms

what happens when you plan a microsaccade to dot3, but just before you execute
your eyes move to dot1?
    a) the ongoing move to dot1 is automatically taken into account so that you
       still move to dot3 as planned
    b) you execute the movement as if you were still in the initial position so
       that you end up in dot1+dot3

events:
      0, 1st dot onset 
    100, 2nd dot onset
    200, 3rd dot onset,
    300, (micro-)saccade to 1st dot position
    300, 4th dot onset
    400, (micro-)saccade to a) 2nd dot position, or b) dot1 + dot2
    400, 5th dot onset
    500, (micro-)saccade to a) 3rd dot position, or b) dot1 + dot2 + dot3
    500, 6th dot onset
    600, (micro-)saccade to aa) dot4 - dot1, or ab) 
    600, 7th dot onset
    700, (micro-)saccade to a) dot5 - dot2, or b) dot5 - (dot1 + dot2)
    700, 8th dot onset
    800, (micro-)saccade to a) dot6 - dot3, or b) dot6 - (dot1 + dot2 + dot3)
    800, 8th dot onset

you have four interacting states:
    1. dot position
    2. eye position
    3. internal belief about eye position used to execute eye movement to 
       desired location
    4. desired location
    
1. dot position changes every 100 ms and only depends on time
2. eye position is assumed to change every 100 ms with a delay of 300 or 400 ms
   and depends on previous eye movements
3. internal belief about eye position changes with each eye movement, but may
   have delay, too
4. the desired location is equal to the dot position
"""

class eye(object):
    @property
    def dotpos(self):
        return self._dotpos
    
    @dotpos.setter
    def dotpos(self, dotpos):
        # dots x (x, y)
        dotpos = np.array(dotpos)
        
        self._dotpos = pd.DataFrame([], index=self.times, columns=['x', 'y'],
                                    dtype=float)
        
        for d in np.arange(dotpos.shape[0]):
            ind = self._dotpos.loc[slice(d * self.dot_dur + self.dt, 
                                         (d + 1) * self.dot_dur)].index
            self._dotpos.loc[ind] = np.tile(dotpos[d, :], (ind.size, 1))
            
    
    def __init__(self, dotpos, dt=10):
        self.dt = dt
        self.times = np.arange(dt, 900, dt)
        
        self.dot_dur = 100
        self.dotonsets = np.arange(0, 900, self.dot_dur)
        
        self.move_delay = 300
        self.move_dur = 50
        
        self.calc_move_times()
        
        self.dotpos = dotpos
        
        self.reset()
        
        
    def reset(self):
        self.move_dir = np.r_[0., 0.]
        self.pos = np.r_[0., 0.]
        self.blfpos = np.r_[0., 0.]
        self.target = deque()
        
    
    def calc_move_times(self):
        mtimes = []
        for do in self.dotonsets:
            times = np.arange(
                    do + self.move_delay + self.dt, 
                    do + self.move_delay + self.move_dur + self.dt, 
                    self.dt)
            mtimes.append(np.c_[times, 
                                do / self.dot_dur * np.ones_like(times)])
        self.move_times = np.concatenate(mtimes)
        
        self.move_onsets = self.dotonsets + self.move_delay
        
        
    def simulate(self):
        self.reset()
        
        out = pd.DataFrame(
                [], dtype=float, index=self.times,
                columns=pd.MultiIndex.from_product(
                        [['pos', 'move_dir', 'target'], ['x', 'y']],
                        names=['state', 'coordinate']))
        
        # handling each time period of [time-dt, time]
        for time in self.times:
            if time in self.dotonsets + self.dt:
                # set currently displayed dot as target for saccade
                self.target.append(self.dotpos.loc[time].values)
            
            if time in self.move_onsets + self.dt:
                # set the movement direction pointing from the believed eye
                # position to the desired target
                self.move_dir = self.target.popleft() - self.blfpos
            
            # if this is a time period during which you move
            if time in self.move_times:
                # move a step towards the current movement direction
                self.pos += self.move_dir / (self.move_dur / self.dt)
            
                self.blfpos = self.pos
            
            out.loc[time, 'target'] = self.target[0]
            out.loc[time, 'pos'] = self.pos
            out.loc[time, 'move_dir'] = self.move_dir
            
        return out