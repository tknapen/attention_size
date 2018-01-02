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



class AttSizeSession(EyelinkSession):

    def __init__(self, *args, **kwargs):

        super(STSession, self).__init__(*args, **kwargs)

        # self.create_screen(full_screen=True, engine='pygaze')

        config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')

        with open(config_file) as config_file:
            config = json.load(config_file)

        self.config = config
        self.create_trials()

        self.stopped = False

    def create_background_stimuli(self):
        ########################################################################################################################
        # Binning over different eccentricities per X and mm
        # Recalculating pixels from the degrees for rings
        ########################################################################################################################

        # Ring positions
        ring_bins = np.digitize(mm_cortex_from_fovea, np.linspace(ecc_min,mm_cortex_from_fovea.max(), conditions_rings, endpoint=False))
        max_ring_positions = np.zeros((len(np.unique(ring_bins)), 1))
        for bi in np.unique(ring_bins):
            max_ring_positions[bi,:] = [ecc_range[ring_bins==bi].max()]


        def degreestopixels_rings(conditions_rings):
            ringposition_degrees = []
            ringposition_pixels = []

            for x in range (0,len(max_ring_positions)):
                deg = max_ring_positions[x,0]
                ringposition_degrees.append(deg)
                pix = (deg*10) * 10.63074
                ringposition_pixels.append(pix)

            return ringposition_degrees, ringposition_pixels

        self.ringposition_degrees, self.ringposition_pixels = degreestopixels_rings(conditions_rings)


        ########################################################################################################################
        # Binning over different eccentricities per X and mm
        # Recalculating pixels from the degrees for blob amounts, locations and sizes
        ########################################################################################################################

        total_blob_rows = (rows_of_blobs * conditions_rings)+3
        blob_bins = np.digitize(mm_cortex_from_fovea, np.linspace(ecc_min,mm_cortex_from_fovea.max(), total_blob_rows, endpoint=False))
        max_blob_positions = np.zeros((len(np.unique(blob_bins)), 1))

        for bi in np.unique(blob_bins):
            max_blob_positions[bi,:] = [ecc_range[blob_bins==bi].max()]

        def degreestopixels_blobs(total_blob_rows):
            blobposition_degrees = []
            blobposition_pixels = []

            for x in range(0, len(max_blob_positions)):       # Minus first and last
                deg = max_blob_positions[x, 0]
                blobposition_degrees.append(deg)
                pix = (deg * 10) * 10.63074
                blobposition_pixels.append(pix)

            return blobposition_degrees, blobposition_pixels

        self.blobposition_degrees, self.blobposition_pixels = degreestopixels_blobs(max_blob_positions)

    def create_trials(self):
        """creates trials by creating a restricted random walk through the display from trial to trial"""

        ########################################################################################################################
        # Cortical Magnification factor & Eccentricities
        ########################################################################################################################

        def cm(ecc, mm_0, tau):
            return mm_0 * np.exp(-ecc*tau)

        mm0 = sym.Symbol('mm0')
        ecc = sym.Symbol('ecc')
        tau = sym.Symbol('tau')
        sym.integrate(sym.exp(-ecc*tau), ecc)

        ecc_nr = 20000
        ecc_range = np.linspace(ecc_min,ecc_max,ecc_nr)
        mm_0 = 12
        tau = 0.5
        fs = 14

        cm_trace = cm(ecc_range, mm_0, tau)
        mm_cortex_from_fovea = np.cumsum(cm_trace) / (ecc_nr/ecc_max)

        ##################################################################################
        ##
        ##  Calculate saccade path for all trials
        ##
        ##################################################################################

        saccade_amplitude_pix = self.deg2pix(self.config['saccade_amplitude'])
        max_eccentricity_pix = self.deg2pix(self.config['max_eccentricity'])
        
        # fixation positions
        present_pos = np.array([0, 0]) # center of screen?        
        trial_saccade_target_positions = [present_pos]

        trial_distractor_positions = [np.array([0, 0])]
        saccade_distractor_directions = [[0,0]]

        for t in range(self.config['n_trials']):
            saccade_direction = np.random.rand() * 2.0 * np.pi
            saccade_vector = np.array([np.sin(saccade_direction)*saccade_amplitude_pix, 
                                        np.cos(saccade_direction)*saccade_amplitude_pix])
            while np.linalg.norm(present_pos + saccade_vector) > max_eccentricity_pix:
                saccade_direction = np.random.rand() * 2.0 * np.pi
                saccade_vector = np.array([np.sin(saccade_direction)*saccade_amplitude_pix, 
                                            np.cos(saccade_direction)*saccade_amplitude_pix])

            which_distractor_direction = np.radians(np.random.choice([-self.config['distractor_deviation_angle'],0,self.config['distractor_deviation_angle']]))
            distractor_vector = np.array([np.sin(saccade_direction + which_distractor_direction)*saccade_amplitude_pix, 
                                            np.cos(saccade_direction + which_distractor_direction)*saccade_amplitude_pix])

            # store this information in list for later use
            trial_saccade_target_positions.append(present_pos + saccade_vector)
            trial_distractor_positions.append(present_pos + distractor_vector)
            saccade_distractor_directions.append([saccade_direction, which_distractor_direction])

            # update present fixation position at the end of the trial
            present_pos = present_pos + saccade_vector

        self.trial_saccade_target_positions = np.array(trial_saccade_target_positions)
        self.trial_distractor_positions = np.array(trial_distractor_positions)
        self.saccade_distractor_directions = np.array(saccade_distractor_directions)


        ##################################################################################
        ##
        ##  And, the stimuli
        ##
        ##################################################################################

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
