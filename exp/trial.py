from __future__ import division
from exptools.core.trial import Trial
import os
import exptools
import json
from psychopy import logging, visual, event
import math
import numpy as np


class AttSizeTrial(Trial):

    def __init__(self, ti, config, stimulus=None, parameters=None, *args, **kwargs):

        self.ID = ti

        if parameters['in_scanner'] == 0:
            if self.ID == 0:
                phase_durations = [1800, parameters['fix_time'], parameters['stim_time'], parameters['post_fix_time']]
            else:
                phase_durations = [parameters['TR'] - (parameters['fix_time'] + parameters['stim_time'] + parameters['post_fix_time']), 
                                    parameters['fix_time'], parameters['stim_time'], parameters['post_fix_time']]
        else:
            phase_durations = [1800, parameters['fix_time'], parameters['stim_time'], parameters['post_fix_time']]

        super(
            AttSizeTrial,
            self).__init__(
            phase_durations=phase_durations,
            *args,
            **kwargs)

        self.parameters = parameters
        # these are placeholders, to be filled in later
        self.parameters['answer'] = -1

        if self.session.config['in_scanner'] == 0:
            self.t_time = self.session.clock.getTime()

        self.session.set_background_stimulus_value(color_balance=self.parameters['bg_trial_stimulus_value'])
        self.session.set_fix_stimulus_value(color_balance=self.parameters['fix_trial_stimulus_value'])

        #print("%f, %f"%(self.parameters['fix_trial_stimulus_value'], self.parameters['bg_trial_stimulus_value']))

    def draw(self, *args, **kwargs):

        # draw additional stimuli:
        if (self.phase == 0 ) * (self.ID == 0):
            self.session.instruction.draw()

        # self.session.draw_prf_stimulus() 
        if (self.parameters['bar_orientation'] != -1) and (self.phase != 0):
            bar_time = self.parameters['pos_in_bar_trajectory'] + (self.session.clock.getTime()-self.t_time)/self.parameters['TR']
            self.session.prf_stim.draw(time=bar_time, 
                                period=self.parameters['prf_stim_barpass_duration'], 
                                orientation=self.parameters['bar_orientation'],
                                position_scaling=1+self.parameters["prf_checker_bar_width"])

        if self.phase == 2:
            self.session.bg_stim.draw()             # surround condition + aperture
        self.session.fixation_disk.draw()       # disk behind fixation condition
        if self.phase == 2:
            self.session.fix_stim.draw()            # fixation condition + aperture

        self.session.fixation_circle.draw()          # circle around fixation condition
        self.session.mask_stim.draw()                # surround aperture
        super(AttSizeTrial, self).draw()


    def event(self):

        for ev in event.getKeys():
            if len(ev) > 0:
                if ev in ['esc', 'escape', 'q']:
                    self.events.append(
                        [-99, self.session.clock.getTime() - self.start_time])
                    self.stop()
                    self.session.stopped = True
                    print 'run canceled by user'
                if ev in ['space', ' ', 't']:
                    if self.phase == 0:
                        if self.ID == 0:
                            self.session.run_time = self.session.clock.getTime()
                        self.t_time = self.session.clock.getTime()
                        self.phase_forward()
                    elif self.phase == 3:
                        self.stop()
                if ev in list(self.session.answer_dictionary.keys()): # staircase answers
                    if self.parameters['answer'] == -1: # only incorporate answer if not answered yet.
                        self.parameters['answer'] = self.session.answer_dictionary[ev]
            print self.parameters['answer']
            super(AttSizeTrial, self).key_event(ev)
