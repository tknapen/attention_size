from psychopy import visual, clock
import numpy as np
import sympy as sym
import os
import json
import glob

import exptools
from exptools.core.session import EyelinkSession
from exptools.core.staircase import ThreeUpOneDownStaircase

from trial import AttSizeTrial
from stim import AttSizeBGStim, PRFStim


class AttSizeSession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(AttSizeSession, self).__init__(*args, **kwargs)

        # self.create_screen(full_screen=True, engine='pygaze')

        config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')

        with open(config_file) as config_file:
            config = json.load(config_file)

        self.config = config
        self.create_trials()

        self.stopped = False

    def create_stimuli(self):
        self.bg_stim = AttSizeBGStim(session=self, 
                        nr_rings=self.config['bg_stim_nr_rings'], 
                        ecc_min=self.config['bg_stim_ecc_min'], 
                        ecc_max=self.config['bg_stim_ecc_max'], 
                        nr_blob_rows_per_ring=self.config['bg_stim_nr_blob_rows_per_ring'], 
                        row_spacing_factor=self.config['bg_stim_ow_spacing_factor'])

        self.prf_stim = PRFStim(session=self, 
                        cycles_in_bar=self.config['prf_checker_cycles_in_bar'], 
                        bar_width=self.config['prf_checker_bar_width'],
                        tex_nr_pix=self.config['prf_checker_tex_nr_pix'],
                        flicker_frequency=self.config['prf_checker_flicker_frequency'])

        size_fixation_pix = self.deg2pix(self.config['size_fixation_deg'])
        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='circle',
                                           size=size_fixation_pix,
                                           texRes=512,
                                           color='white',
                                           sf=0,
                                           pos=(0,0))
        
        this_instruction_string = """Follow the red dot with your gaze. Press 'space' to start. """
        self.instruction = visual.TextStim(self.screen, 
            text = this_instruction_string, 
            font = 'Helvetica Neue',
            pos = (0, 0),
            italic = True, 
            height = 20, 
            alignHoriz = 'center',
            color=(1,0,0))
        self.instruction.setSize((1200,50))


    def create_trials(self):
        """creates trials by setting up staircases for background task, and prf stimulus sequence"""

        ##################################################################################
        ##
        ##################################################################################

        for 



    def run(self):
        """run the session"""
        # cycle through trials

        for ti in range(self.config['n_trials']):

            parameters = {
                'fix_x': self.trial_saccade_target_positions[ti,0],
                'fix_y': self.trial_saccade_target_positions[ti,1],
                'distractor_x': self.trial_distractor_positions[ti,0],
                'distractor_y': self.trial_distractor_positions[ti,1],
                'saccade_direction': self.saccade_distractor_directions[ti,0],
                'distractor_direction': self.saccade_distractor_directions[ti,1],
            }

            parameters.update(self.config)

            if ti == 0:
                parameters['fixation_time'] = 30.0

            trial = AttSizeTrial(ti=ti,
                           config=self.config,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run()

            if self.stopped == True:
                break

        self.close()
