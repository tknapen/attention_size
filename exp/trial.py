from __future__ import division
from exptools.core.trial import Trial
import os
import exptools
import json
from psychopy import logging, visual, event
import numpy as np


class AttSizeTrial(Trial):

    def __init__(self, ti, config, stimulus=None, parameters=None, *args, **kwargs):

        self.ID = ti

        if self.ID == 0:
            phase_durations = [180, parameters['fix_time'], parameters['stim_time'], parameters['post_fix_time']]
        else:
            phase_durations = [-0.0001, parameters['fix_time'], parameters['stim_time'], parameters['post_fix_time']]

        super(
            AttSizeTrial,
            self).__init__(
            phase_durations=phase_durations,
            *args,
            **kwargs)

        self.parameters = parameters
        # these are placeholders, to be filled in later
        self.parameters['answer'] = -1
        self.parameters['staircase_value'] = 0
        self.parameters['correct'] = -1

        self.session.set_background_stimulus_values()

    def draw(self, *args, **kwargs):

        # draw additional stimuli:
        if (self.phase == 0 ) * (self.ID == 0):
            self.session.instruction.draw()

        if self.index_number == 1:
            self.session.draw_prf_stimulus() 

        if self.phase == 2:
            self.session.bg_stim.draw()             # surround condition + aperture
            self.session.fixation_disk.draw()     # circle around fixation condition
            self.session.fix_stim.draw()            # fixation condition + aperture

        self.session.fixation_circle.draw()     # circle around fixation condition
        self.session.mask_stim.draw()           # surround aperture
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
                    if self.phase == 0 and self.ID == 0:
                        self.session.run_time = self.session.clock.getTime()
                        self.phase_forward()
                    elif self.phase == 3:
                        self.stop()
                if ev in list(self.session.answer_dictionary.keys()): # staircase answers
                    if self.parameters['answer'] == -1: # only incorporate answer if not answered yet.
                        # and, only incorporate answer into staircase pertaining to present task
                        if self.session.task == 0: # fixation
                            self.parameters['staircase_value'] = self.session.fix_staircase.get_intensity()
                        elif self.session.task == 1: # surround
                            self.parameters['staircase_value'] = self.session.bg_staircase.get_intensity()

                        self.parameters['answer'] = self.session.answer_dictionary[ev]
                        self.parameters['correct'] = int(self.session.answer_dictionary[ev] == self.session.which_correct_answer)


                        ##### temp print stuff #####
                        print ('---------------------------------')
                        print('key answered %i '%(self.parameters['answer']))
                        print('correct was %i '%self.session.which_correct_answer)

                        if self.parameters['correct'] == 0:
                            print ('> trial outcome was incorrect')
                        
                        else:  
                            print ('> trial outcome was correct')
                        ##############################
                        

                        # incorporate answer
                        if self.session.task == 0: # fixation
                            self.session.fix_staircase.answer(correct=self.parameters['correct'])
                        elif self.session.task == 1: # surround
                            self.session.bg_staircase.answer(correct=self.parameters['correct'])
            super(AttSizeTrial, self).key_event(ev)
