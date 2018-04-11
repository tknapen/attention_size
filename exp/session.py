from __future__ import division
from psychopy import visual, clock
from psychopy import filters
import numpy as np
import sympy as sym
import os
import json
import glob
import copy
import scipy
import pickle

import exptools
from exptools.core.session import EyelinkSession
from exptools.core.staircase import ThreeUpOneDownStaircase, OneUpOneDownStaircase

from trial import AttSizeTrial
from stim import AttSizeBGStim, AttSizeBGPixelFest, PRFStim


class AttSizeSession(EyelinkSession):

    def __init__(self, subject_initials, index_number, task, tracker_on, *args, **kwargs):

        super(AttSizeSession, self).__init__(subject_initials, index_number, tracker_on, *args, **kwargs)

        self.create_screen(full_screen=False, engine='pygaze')
        self.task = task
        config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')

        with open(config_file) as config_file:
            config = json.load(config_file)

        self.config = config

        # dict that maps button keys to answers
        self.answer_dictionary =  self.config['answer_dictionary']     # NEW JR       

        self.create_stimuli()
        self.create_trials()

        self.run_time = -1
        self.stopped = False

    def create_stimuli(self):

        self.prf_stim = PRFStim(session=self, 
                        cycles_in_bar=self.config['prf_checker_cycles_in_bar'], 
                        bar_width=self.config['prf_checker_bar_width'],
                        tex_nr_pix=self.config['prf_checker_tex_nr_pix'],
                        flicker_frequency=self.config['prf_checker_flicker_frequency'],
                        max_eccentricity=self.config['blobstim_bg_ecc_max']*self.pixels_per_degree)     # NEW JR

        size_fixation_pix = self.deg2pix(self.config['size_fixation_deg'])
        self.fixation = visual.GratingStim(self.screen,
                                           tex='sin',
                                           mask='circle',
                                           size=size_fixation_pix,
                                           texRes=512,
                                           color='white',
                                           sf=0,
                                           pos=(0,0))
        
        if self.task == 0:  # NEW JR 
            if self.config['which_stimulus_type'] == 0: # blobs
                this_instruction_string = """Fixate in the center of the screen. Your task is to judge whether the blobs in the SMALL circle are more red or more green on every stimulus presentation.\n\n> Press left for red\n> Press right for green"""
        
            if self.config['which_stimulus_type'] == 1: # pixelstim
                this_instruction_string = """Fixate in the center of the screen. Your task is to judge whether the pixels in the SMALL circle are more red or more green on every stimulus presentation.\n\n> Press left for red\n> Press right for green"""
        
        elif self.task == 1:  # NEW JR 
            if self.config['which_stimulus_type'] == 0: # blobs
                this_instruction_string = """Fixate in the center of the screen. Your task is to judge whether the blobs in the ENTIRE circle are more red or more green on every stimulus presentation.\n\n> Press left for red\n> Press right for green"""
        
            if self.config['which_stimulus_type'] == 1: # pixelstim
                this_instruction_string = """Fixate in the center of the screen. Your task is to judge whether the pixels in the ENTIRE circle are more red or more green on every stimulus presentation.\n\n> Press left for red\n> Press right for green"""

        self.instruction = visual.TextStim(self.screen, 
            text = this_instruction_string, 
            font = 'Helvetica Neue',
            pos = (0, 0),
            italic = True, 
            height = 30, 
            alignHoriz = 'center',
            color=(-1,-1,1))
        self.instruction.setSize((500,150))

        mask = filters.makeMask(matrixSize=self.screen_pix_size[0], 
                                shape='raisedCosine', 
                                radius=self.config['pixstim_bg_aperture_fraction']*(self.screen_pix_size[1]/self.screen_pix_size[0]),
                                center=(0.0, 0.0), 
                                range=[-1, 1], 
                                fringeWidth=self.config['aperture_stim_sd']*(self.screen_pix_size[1]/self.screen_pix_size[0])
                                )

        self.mask_stim = visual.PatchStim(self.screen, 
                                        mask=-mask, 
                                        tex=None, 
                                        size=[self.screen_pix_size[0],self.screen_pix_size[0]], 
                                        pos = np.array((0.0,0.0)), 
                                        color = (-1,-1,-1))# self.screen.background_color) 

        # Small fixation condition ring
        self.fixation_circle = visual.Circle(self.screen, 
            radius=self.config['size_fixation_deg']*self.pixels_per_degree/2, 
            units='pix', lineColor='black')
        self.fixation_disk = visual.Circle(self.screen, 
            units='pix', radius=self.config['size_fixation_deg']*self.pixels_per_degree/2, 
            fillColor=self.screen.background_color)



        if self.config['which_stimulus_type'] == 0: # blob surround stimulus    # NEW JR
            self.bg_stim = AttSizeBGStim(session=self, 
                        nr_rings=self.config['blobstim_bg_nr_rings'], 
                        ecc_min=self.config['blobstim_bg_ecc_min'], 
                        ecc_max=self.config['blobstim_bg_ecc_max'], 
                        nr_blob_rows_per_ring=self.config['blobstim_bg_nr_blobrows_per_ring'], 
                        row_spacing_factor=self.config['blobstim_bg_spacing_factor'],
                        opacity=self.config['blobstim_bg_opacity'])


            self.fix_stim = AttSizeBGStim(session=self,  # blob fixation stimulus    # NEW JR
                        nr_rings=self.config['blobstim_fix_nr_ring'], 
                        ecc_min=self.config['blobstim_fix_ecc_min'], 
                        ecc_max=self.config['size_fixation_deg'], 
                        nr_blob_rows_per_ring=self.config['blobstim_fix_nr_blobrows_per_ring'], 
                        row_spacing_factor=self.config['blobstim_fix_spacing_factor'],
                        opacity=self.config['blobstim_fix_opacity'])


        elif self.config['which_stimulus_type'] == 1: # 1/f noise surround stimulus
            self.bg_stim = AttSizeBGPixelFest(session=self,
                        tex_size=self.config['pixstim_bg_tex_size'],
                        amplitude_exponent=self.config['pixstim_bg_amplitude_exponent'], 
                        n_textures=self.config['pixstim_bg_noise_n_textures'],
                        opacity=self.config['pixstim_bg_opacity'],
                        size=self.screen.size[1],
                        aperture_fraction=self.config['pixstim_bg_aperture_fraction'])                
        
            self.fix_stim = AttSizeBGPixelFest(session=self, # 1/f noise fixation stimulus
                        tex_size=self.config['pixstim_fix_tex_size'],
                        amplitude_exponent=self.config['pixstim_fix_amplitude_exponent'], 
                        n_textures=self.config['pixstim_fix_noise_n_textures'],
                        opacity=self.config['pixstim_fix_opacity'],
                        size=(1.0/self.config['pixstim_fix_aperture_fraction']) * self.config['size_fixation_deg']*self.pixels_per_degree,
                        aperture_fraction=self.config['pixstim_fix_aperture_fraction'])     


    def create_trials(self):
        """creates trials by setting up staircases for background task, and prf stimulus sequence"""

        ##################################################################################
        ## timings etc for the bar passes
        ##################################################################################

        self.prf_bar_pass_times = np.r_[0, np.cumsum(np.array([self.config['prf_stim_barpass_duration']*self.config['TR'] 
                                        for prf_ori in self.config['prf_stim_sequence_angles']])), 1e7]

        
        self.stimulus_values = np.unique(np.r_[np.linspace(0,self.config['psychometric_curve_interval'], self.config['psychometric_curve_nr_steps_oneside']),
                                    -np.linspace(0,self.config['psychometric_curve_interval'], self.config['psychometric_curve_nr_steps_oneside'])])

        self.nr_trials = self.config['prf_stim_barpass_duration'] * len(self.config['prf_stim_sequence_angles'])
        self.fix_trial_stimulus_values = np.random.choice(self.stimulus_values, self.nr_trials)
        self.bg_trial_stimulus_values = np.random.choice(self.stimulus_values, self.nr_trials)


##################################################################################
## Fixation conditions - value staircases
##################################################################################

    def set_fix_stimulus_value(self, color_balance=0):
        """set_fix_stimulus_values sets the value of the fixation stimulus based on the fix_staircase.
        It has to take into account the stimulus type and type of staircase to construct the values,
        and will only set the self.which_correct_answer if the fixation task is being performed in this run.
        """

        ##################################################################################
        ## Blobs OneUpOneDown Staircase - fixation
        ##################################################################################

        if self.config['which_stimulus_type'] == 0: # blobs
            self.fix_stim.repopulate_condition_ring_colors(condition_nr=0, 
                color_balance=color_balance)

        ##################################################################################
        ## Noisestim OneUpOneDown Staircase - fixation
        ##################################################################################

        elif self.config['which_stimulus_type'] == 1: # 1/f noise
            self.fix_stim.recalculate_stim(  red_boost=-color_balance,     # added JR
                                                green_boost=color_balance,     # added JR
                                                blue_boost=0)                   # added JR


##################################################################################
## Surround conditions - value staircases
##################################################################################

    def set_background_stimulus_value(self, color_balance):
        """set_background_stimulus_values sets the value of the fixation stimulus based on the fix_staircase.
        It has to take into account the stimulus type and type of staircase to construct the values,
        and will only set the self.which_correct_answer if the background task is being performed in this run.
        """

        ##################################################################################
        ## Blobs OneUpOneDown Staircase - surround
        ##################################################################################

        if self.config['which_stimulus_type'] == 0: # blobs

            self.bg_stim.repopulate_condition_ring_colors(condition_nr=0, 
                color_balance=color_balance)

        ##################################################################################
        ## Noisestim OneUpOneDown Staircase - surround
        ##################################################################################

        elif self.config['which_stimulus_type'] == 1: # 1/f noise
            self.bg_stim.recalculate_stim(  red_boost=-color_balance,
                                                green_boost=color_balance, 
                                                blue_boost=0)    


    def draw_prf_stimulus(self):
        # only draw anything after the experiment has started
        if self.run_time > 0:
            present_time = self.clock.getTime() - self.run_time
            present_bar_pass = np.arange(len(self.prf_bar_pass_times))[(self.prf_bar_pass_times - present_time)>0][0]-1
            prf_time = present_time - self.prf_bar_pass_times[present_bar_pass]
            if self.config['prf_stim_sequence_angles'][present_bar_pass] != -1:
                self.prf_stim.draw(time=prf_time/self.config['TR'], 
                                period=self.config['prf_stim_barpass_duration'], 
                                orientation=self.config['prf_stim_sequence_angles'][present_bar_pass],
                                position_scaling=1+self.config["prf_checker_bar_width"])

    def close(self):
        super(AttSizeSession, self).close()

    def run(self):
        """run the session"""
        # cycle through trials

        self.ti = 0
        while not self.stopped:

            parameters = copy.copy(self.config)
            parameters['fix_trial_stimulus_value'] = self.fix_trial_stimulus_values[self.ti]
            parameters['bg_trial_stimulus_value'] = self.bg_trial_stimulus_values[self.ti]

            trial = AttSizeTrial(ti=self.ti,
                           config=self.config,
                           screen=self.screen,
                           session=self,
                           parameters=parameters,
                           tracker=self.tracker)
            trial.run()
            self.ti += 1

            if self.stopped == True:
                break

        self.close()
