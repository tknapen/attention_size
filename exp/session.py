from psychopy import visual, clock
from psychopy import filters
import numpy as np
import sympy as sym
import os
import json
import glob
import copy
import scipy

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
        self.create_stimuli()
        self.create_trials()

        # dict that maps button keys to answers
        self.answer_dictionary = {'b': 0, 'g': 1}

        self.run_time = -1
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

        mask = filters.makeMask(matrixSize=self.screen_pix_size[0], 
                                shape='raisedCosine', 
                                radius=self.config['aperture_max_eccen'], 
                                # center=(0.0, 0.0), 
                                # range=[1, -1], 
                                fringeWidth=self.config['aperture_sd']
                                )
        self.mask_stim = visual.PatchStim(self.screen, 
                                        mask=-mask, 
                                        tex=None, 
                                        size=[self.screen_pix_size[0],self.screen_pix_size[0]], 
                                        pos = np.array((0.0,0.0)), 
                                        color = self.screen.background_color) 

    def create_trials(self):
        """creates trials by setting up staircases for background task, and prf stimulus sequence"""

        ##################################################################################
        ## timings etc for the bar passes
        ##################################################################################

        self.prf_bar_pass_times = np.r_[0,np.cumsum(np.array([self.config['prf_stim_barpass_duration'] 
                                        for prf_ori in self.config['prf_stim_sequence_angles']]))]
        print(self.prf_bar_pass_times)
        ##################################################################################
        ## staircases
        ##################################################################################

        self.staircase_file = os.path.join('data', self.subject_initials + '_' + str(self.index_number) + '.pkl')
        if os.path.isfile(self.staircase_file):
            with open(self.staircase_file, 'a') as f:
                self.staircase = pickle.load(f)
        else:
            self.staircase = ThreeUpOneDownStaircase(initial_value=self.config['staircase_initial_value'], 
                                                initial_stepsize=self.config['staircase_initial_value']/4.0, 
                                                nr_reversals = 3000, 
                                                increment_value = self.config['staircase_initial_value']/4.0, 
                                                stepsize_multiplication_on_reversal = 0.9, 
                                                max_nr_trials = 12000, 
                                                min_test_val = None, 
                                                max_test_val = 0.5)
        self.set_background_stimulus_values()

    def set_background_stimulus_values(self):
        this_intensity = self.staircase.get_intensity()

        for ring in np.arange(self.config['bg_stim_nr_rings']):
            correct_answer_this_ring = np.random.randint(0,2)
            this_ring_color_balance = 0.5 + ((correct_answer_this_ring*2)-1) * this_intensity
            self.bg_stim.repopulate_condition_ring_colors(condition_nr=ring,
                                                            color_balance=this_ring_color_balance)
            if ring == self.index_number: # this is the ring for which the answers are recorded
                self.which_correct_answer = correct_answer_this_ring
                self.signal_ring_color_balance = this_ring_color_balance

    def draw_prf_stimulus(self):
        # only draw anything after the experiment has started
        if self.run_time > 0:
            present_time = self.clock.getTime() - self.run_time
            present_bar_pass = np.arange(len(self.prf_bar_pass_times))[(self.prf_bar_pass_times - present_time)>0][0]-1
            prf_time = present_time - self.prf_bar_pass_times[present_bar_pass]
            # print(present_time, present_bar_pass, prf_time)
            if self.config['prf_stim_sequence_angles'][present_bar_pass] != -1:
                self.prf_stim.draw(time=prf_time, 
                                period=self.config['prf_stim_barpass_duration'], 
                                orientation=self.config['prf_stim_sequence_angles'][present_bar_pass])

    def run(self):
        """run the session"""
        # cycle through trials

        ti = 0
        while not self.stopped:

            parameters = copy.copy(self.config)

            trial = AttSizeTrial(ti=ti,
                           config=self.config,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run()
            ti += 1

            if self.stopped == True:
                break

        self.close()
