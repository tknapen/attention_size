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
        ## timings etc for the bar passes
        ##################################################################################

        self.prf_bar_pass_times = np.cumsum(np.array([self.config['prf_stim_barpass_duration'] 
                                        for prf_ori in self.config['prf_stim_sequence_angles']]))


        ##################################################################################
        ## staircases
        ##################################################################################

        self.staircase_file = os.path.join('data', self.initials + '_' + str(index_number) + '.pkl')
        if os.path.isfile(self.staircase_file):
            with open(self.staircase_file, 'a') as f:
                self.staircase = pickle.load(f)
        else:
            self.staircase = ThreeUpOneDownStaircase(initial_value=self.config['staircase_initial_value'], 
                                                initial_stepsize=self.config['staircase_initial_value']/4.0, 
                                                nr_reversals = 3000, 
                                                increment_value = self.config['staircase_initial_value']/4.0, 
                                                stepsize_multiplication_on_reversal = 0.9, 
                                                max_nr_trials = 12000 , 
                                                min_test_val = None, 
                                                max_test_val = 0.5)

    def set_background_stimulus_values(self):
        this_intensity = self.staircase.get_intensity()

        for ring in np.arange(self.config['nr_ring_conditions']):
            correct_answer_this_ring = np.random.randint(2,1)
            this_ring_color_balance = 0.5 + ((correct_answer_this_ring*2)-1) * this_intensity
            self.bg_stim.repopulate_condition_ring_colors(condition_nr=ring,
                                                            color_balance=this_ring_color_balance)
            if ring == self.index_number: # this is the ring for which the answers are recorded
                self.which_correct_answer = correct_answer_this_ring


    def run(self):
        """run the session"""
        # cycle through trials

        for ti in range(self.config['n_trials']):


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
