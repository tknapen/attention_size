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

    def __init__(self, subject_initials, index_number, task, baseline_RG_this_subject, tracker_on, *args, **kwargs):

        super(AttSizeSession, self).__init__(subject_initials, index_number, tracker_on, *args, **kwargs)

        self.create_screen(full_screen=False, engine='pygaze')
        self.task = task
        self.baseline_RG_this_subject = baseline_RG_this_subject
        config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')

        with open(config_file) as config_file:
            config = json.load(config_file)

        self.config = config

        # dict that maps button keys to answers
        #self.answer_dictionary = {'g': 0, 'b': 1} 
        self.answer_dictionary =  self.config['answer_dictionary']     # NEW JR       

        # construct name for staircase file
        self.fix_staircase_file = os.path.join('data', self.subject_initials + '_' + str(self.index_number) + '_0.pkl')
        self.bg_staircase_file = os.path.join('data', self.subject_initials + '_' + str(self.index_number) + '_1.pkl')

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

        # if self.task == 0:
        #     this_instruction_string = """Fixate in the center of the screen. 
        #         Your task is to judge whether the fixation marker 
        #         is more red or more green on every stimulus presentation. """
        # elif self.task == 1:
        #     this_instruction_string = """Fixate in the center of the screen.  
        #         Your task is to judge whether the background 
        #         is more red or more green on every stimulus presentation. """

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

        ##################################################################################
        ## fixation staircases
        ##################################################################################
        if os.path.isfile(self.fix_staircase_file):
            with open(self.fix_staircase_file, 'r') as f:
                self.fix_staircase = pickle.load(f)
        else:
            if self.index_number == 0: 
                self.fix_staircase = OneUpOneDownStaircase(initial_value=self.config['1u1d_center_staircase_initial_value'], 
                                            initial_stepsize=self.config['1u1d_center_staircase_initial_value']/4.0, 
                                            nr_reversals = 3000, 
                                            increment_value = self.config['1u1d_center_staircase_initial_value']/4.0, 
                                            stepsize_multiplication_on_reversal = 0.8, 
                                            max_nr_trials = 12000, 
                                            min_test_val = None, 
                                            max_test_val = 0.5)

            elif self.index_number == 1:
                self.fix_staircase = ThreeUpOneDownStaircase(initial_value=self.config['3u1d_center_staircase_initial_value'], 
                                                initial_stepsize=self.config['3u1d_center_staircase_initial_value']/4.0, 
                                                nr_reversals = 3000, 
                                                increment_value = self.config['3u1d_center_staircase_initial_value']/4.0, 
                                                stepsize_multiplication_on_reversal = 0.9, 
                                                max_nr_trials = 12000, 
                                                min_test_val = None, 
                                                max_test_val = 0.5)


        ##################################################################################
        ## surround staircases
        ##################################################################################

        if os.path.isfile(self.bg_staircase_file):
            with open(self.bg_staircase_file, 'r') as f:
                self.bg_staircase = pickle.load(f)
        else:
            if self.index_number == 0: 
                self.bg_staircase = OneUpOneDownStaircase(initial_value=self.config['1u1d_surround_staircase_initial_value'], 
                                            initial_stepsize=self.config['1u1d_surround_staircase_initial_value']/4.0, 
                                            nr_reversals = 3000, 
                                            increment_value = self.config['1u1d_surround_staircase_initial_value']/4.0, 
                                            stepsize_multiplication_on_reversal = 0.8, 
                                            max_nr_trials = 12000, 
                                            min_test_val = None, 
                                            max_test_val = 0.5)

            elif self.index_number == 1:
                self.bg_staircase = ThreeUpOneDownStaircase(initial_value=self.config['3u1d_surround_staircase_initial_value'], 
                                                initial_stepsize=self.config['3u1d_surround_staircase_initial_value']/4.0, 
                                                nr_reversals = 3000, 
                                                increment_value = self.config['3u1d_surround_staircase_initial_value']/4.0, 
                                                stepsize_multiplication_on_reversal = 0.9, 
                                                max_nr_trials = 12000, 
                                                min_test_val = None, 
                                                max_test_val = 0.5)
 
        self.set_stimulus_values()

    def set_stimulus_values(self):

        self.set_background_stimulus_values()
        self.set_fix_stimulus_values()



##################################################################################
## Fixation conditions - value staircases
##################################################################################

    def set_fix_stimulus_values(self):
        """set_fix_stimulus_values sets the value of the fixation stimulus based on the fix_staircase.
        It has to take into account the stimulus type and type of staircase to construct the values,
        and will only set the self.which_correct_answer if the fixation task is being performed in this run.
        """

        this_intensity = self.fix_staircase.get_intensity()

        ##################################################################################
        ## Blobs OneUpOneDown Staircase - fixation
        ##################################################################################

        if self.config['which_stimulus_type'] == 0: # blobs
            if self.index_number == 0: # one up one down staircase goes only from one direction
                which_correct_answer = 0

        ##################################################################################
        ## Blobs ThreeUpOneDown Staircase - fixation
        ##################################################################################    

            elif self.index_number == 1: # three up one down staircase
                which_correct_answer = np.random.randint(0,2)
 
            color_balance = 0.5 + ((which_correct_answer*2)-1) * this_intensity   
            self.fix_stim.repopulate_condition_ring_colors(condition_nr=0, 
                color_balance=color_balance)

        ##################################################################################
        ## Noisestim OneUpOneDown Staircase - fixation
        ##################################################################################

        elif self.config['which_stimulus_type'] == 1: # 1/f noise
            if self.index_number == 0: # one up one down staircase goes only from one direction
                which_correct_answer = 0                                        # added JR
                answer_sign = (which_correct_answer*2)-1                        # added JR
                self.fix_stim.recalculate_stim(  red_boost=-this_intensity,     # added JR
                                                green_boost=this_intensity,     # added JR
                                                blue_boost=0)                   # added JR

        ##################################################################################
        ## Noisestim ThreeUpOneDown Staircase - fixation
        ##################################################################################

            elif self.index_number == 1: # three up one down staircase
                which_correct_answer = np.random.randint(0,2)
                answer_sign = (which_correct_answer*2)-1
                self.fix_stim.recalculate_stim(  red_boost=self.baseline_RG_this_subject-(this_intensity*answer_sign),     # added JR
                                                green_boost=self.baseline_RG_this_subject+(this_intensity*answer_sign),    # added JR
                                                blue_boost=0)

        # regardless of stimulus type, we now know the correct answer.
        if self.task == 0: # subject is actually doing fixation task
            self.which_correct_answer = which_correct_answer
            print('fix stim: ca %i, int %f '%(self.which_correct_answer, this_intensity))   


##################################################################################
## Surround conditions - value staircases
##################################################################################

    def set_background_stimulus_values(self):
        """set_background_stimulus_values sets the value of the fixation stimulus based on the fix_staircase.
        It has to take into account the stimulus type and type of staircase to construct the values,
        and will only set the self.which_correct_answer if the background task is being performed in this run.
        """

        this_intensity = self.bg_staircase.get_intensity()

        ##################################################################################
        ## Blobs OneUpOneDown Staircase - surround
        ##################################################################################

        if self.config['which_stimulus_type'] == 0: # blobs
            if self.index_number == 0: # one up one down staircase goes only from one direction
                which_correct_answer = 0

        ##################################################################################
        ## Blobs ThreeUpOneDown Staircase - surround
        ##################################################################################
                
            elif self.index_number == 1: # three up one down staircase
                which_correct_answer = np.random.randint(0,2)
 
            color_balance = 0.5 + ((which_correct_answer*2)-1) * this_intensity   
            self.bg_stim.repopulate_condition_ring_colors(condition_nr=0, 
                color_balance=color_balance)

        ##################################################################################
        ## Noisestim OneUpOneDown Staircase - surround
        ##################################################################################

        elif self.config['which_stimulus_type'] == 1: # 1/f noise
            if self.index_number == 0: # one up one down staircase goes only from one direction
                which_correct_answer = 0
                answer_sign = (which_correct_answer*2)-1
                self.bg_stim.recalculate_stim(  red_boost=-this_intensity,
                                                green_boost=this_intensity, 
                                                blue_boost=0)    

        ##################################################################################
        ## Noisestim ThreeUpOneDown Staircase - surround
        ##################################################################################
                
            elif self.index_number == 1: # three up one down staircase
                which_correct_answer = np.random.randint(0,2)
                answer_sign = (which_correct_answer*2)-1
                self.bg_stim.recalculate_stim(  red_boost=self.baseline_RG_this_subject-(this_intensity*answer_sign),
                                                green_boost=self.baseline_RG_this_subject+(this_intensity*answer_sign), 
                                                blue_boost=0)

        # regardless of stimulus type, we now know the correct answer.
        if self.task == 1: # subject is actually doing bg task
            self.which_correct_answer = which_correct_answer
            print('bg stim: ca %i, int %f '%(self.which_correct_answer, this_intensity*-answer_sign))



    def draw_prf_stimulus(self):
        # only draw anything after the experiment has started
        if self.run_time > 0:
            present_time = self.clock.getTime() - self.run_time
            present_bar_pass = np.arange(len(self.prf_bar_pass_times))[(self.prf_bar_pass_times - present_time)>0][0]-1
            prf_time = present_time - self.prf_bar_pass_times[present_bar_pass]
            # print(present_time, present_bar_pass, prf_time)
            if self.config['prf_stim_sequence_angles'][present_bar_pass] != -1:
                self.prf_stim.draw(time=prf_time/self.config['TR'], 
                                period=self.config['prf_stim_barpass_duration'], 
                                orientation=self.config['prf_stim_sequence_angles'][present_bar_pass],
                                position_scaling=1+self.config["prf_checker_bar_width"])

    def close(self):
        # print(self.outputDict['eventArray'])
        with open(self.bg_staircase_file, 'w') as f:
            pickle.dump(self.bg_staircase, f)
        with open(self.fix_staircase_file, 'w') as f:
            pickle.dump(self.fix_staircase, f)

        super(AttSizeSession, self).close()

    def run(self):
        """run the session"""
        # cycle through trials

        self.ti = 0
        while not self.stopped:

            parameters = copy.copy(self.config)
            parameters['baseline_RG'] = self.baseline_RG_this_subject

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
