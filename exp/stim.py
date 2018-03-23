from __future__ import division
from exptools.core.session import EyelinkSession
from trial import AttSizeTrial
from psychopy import visual, clock, filters
import numpy as np
import sympy as sym
from math import * 
import os
import exptools
import json
import glob
from scipy import ndimage

# from session import AttSizeSession


def ecc_deg_2_mm(eps, lmb=1.2, eps0=1.0):
    """Returns X, the 'eccentricity' in cm"""
    return lmb * np.log(1 + (eps/eps0))

def ecc_mm_2_deg(X, lmb=1.2, eps0=1.0):
    """Returns epsilon, the eccentricity in degrees"""
    return -eps0 + np.exp(X/lmb)


# Polarcoordinate function
def pol2cart(rho, phi):
    x = rho * cos(phi)
    y = rho * sin(phi)
    return(x, y)


# Calculate arc of blob, related to circle radius
def line2arc(radius,distance):
     return 2*radius*asin(distance/(2*radius))


def PointsInCircum(r,n):
    return [(cos(2*pi/n*x)*r,sin(2*pi/n*x)*r) for x in xrange(0,n+1)]


# Math thingy for substracting lists from lists
class MyList(list):
    def __init__(self, *args):
        super(MyList, self).__init__(args)

    def __sub__(self, other):
        return self.__class__(*[item for item in self if item not in other])

########################################################################################################################
# Cortical Magnification factor & Eccentricities
########################################################################################################################

def cm(ecc, mm_0, tau):
    return mm_0 * np.exp(-ecc*tau)

def mm_cortex_from_fovea(ecc_min, ecc_max, nr=1e4, mm_0=12, tau=0.5, fs=14):
    mm_range = np.linspace(ecc_min,ecc_max,nr)

    ecc_range = ecc_min + ecc_max*np.exp(mm_range)/np.exp(ecc_max)

    cm_trace = cm(mm_range, mm_0, tau)
    mm_cortex_from_fovea = np.cumsum(cm_trace) / (nr/ecc_max) 

    return mm_cortex_from_fovea, mm_range, cm_trace


class AttSizeBGStim(object):
    def __init__(self, 
                session, 
                nr_rings=3, 
                ecc_min=0.1, 
                ecc_max=5, 
                nr_blob_rows_per_ring=24, 
                row_spacing_factor=0.8, 
                opacity=0.125,
                **kwargs):
        
        self.session = session
        self.opacity = opacity
        total_nr_rows = nr_rings*(nr_blob_rows_per_ring+1)+1

        ecc_min_mm, ecc_max_mm = ecc_deg_2_mm(ecc_min), ecc_deg_2_mm(ecc_max)

        rows_ecc_in_mm = np.linspace(ecc_min_mm, ecc_max_mm, total_nr_rows, endpoint = True)
        rows_ecc_in_deg = ecc_mm_2_deg(rows_ecc_in_mm)

        # Ring positions
        ring_pos = np.arange(0, total_nr_rows, nr_blob_rows_per_ring+1)
        blob_row_pos = np.setdiff1d(np.arange(total_nr_rows), ring_pos)
        blob_row_sizes = np.diff(rows_ecc_in_deg)
        blob_row_blob_sizes = blob_row_sizes * row_spacing_factor

        nr_blobs_per_ring = ((2*np.pi*rows_ecc_in_deg[1:])/blob_row_sizes).astype(int)

        element_array_np = []
        ring_nr = 0
        for ecc, bpr, s in zip(rows_ecc_in_deg[:], nr_blobs_per_ring[:], blob_row_blob_sizes[:]):
            if not ring_nr in ring_pos:            
                ring_condition = floor(nr_rings * ring_nr/total_nr_rows)
                for pa in np.linspace(0, 2*np.pi, bpr, endpoint=False):
                    x, y = pol2cart(ecc, pa)
                    element_array_np.append([self.session.pixels_per_degree*x, 
                                            self.session.pixels_per_degree*y, 
                                            ecc, 
                                            pa, 
                                            self.session.pixels_per_degree*s,
                                            1, 1, 1, 0.2, ring_nr, ring_condition])
            ring_nr += 1

        self.element_array_np = np.array(element_array_np)

        self.element_array_stim = visual.ElementArrayStim(self.session.screen, 
                                                    colors=self.element_array_np[:,[5,6,7]], 
                                                    colorSpace='rgb', 
                                                    nElements=self.element_array_np.shape[0], 
                                                    sizes=self.element_array_np[:,4], 
                                                    sfs=0, 
                                                    opacities=self.opacity,
                                                    xys=self.element_array_np[:,[0,1]])
        for i in range(nr_rings):
            self.repopulate_condition_ring_colors(condition_nr=i, color_balance=0.5)

        # rings are easy, blobs are hard
        self.rings = [visual.Circle(self.session.screen, 
                                            radius=rows_ecc_in_deg[rp]*self.session.pixels_per_degree, 
                                            lineColor=[-1, -1, -1], 
                                            edges=256) for rp in ring_pos]


    def repopulate_condition_ring_colors(self, condition_nr, color_balance):
        this_ring_bool = self.element_array_np[:,-1] == condition_nr
        nr_elements_in_condition = this_ring_bool.sum()

        nr_signal_elements = int(nr_elements_in_condition * color_balance)
        ordered_signals = np.r_[np.ones(nr_signal_elements), -np.ones(nr_elements_in_condition-nr_signal_elements)]
        np.random.shuffle(ordered_signals)

        self.element_array_np[this_ring_bool, 5] = ordered_signals
        self.element_array_np[this_ring_bool, 6] = -ordered_signals

        self.insert_parameters_into_stim()

    def insert_parameters_into_stim(self):
        self.element_array_stim.setColors(self.element_array_np[:,[5,6,7]], log=False)
        self.element_array_stim.setXYs(self.element_array_np[:,[0,1]], log=False)

    def recalculate_elements(self):
        pass

    def draw(self):
        self.element_array_stim.draw()

    def draw_circles(self):
        for ring in self.rings:
            ring.draw()


class AttSizeBGPixelFest(object):
    def __init__(self, 
                session,
                tex_size,
                amplitude_exponent, 
                n_textures,
                opacity,
                size,            # NEW
                aperture,        # NEW
                **kwargs):  

        self.session = session
        self.tex_size = tex_size
        self.amplitude_exponent = amplitude_exponent
        self.n_textures = n_textures
        self.opacity = opacity
        self.size = size                            # NEW
        self.aperture = aperture                    # NEW


        self.basic_textures = self.create_basic_textures(self.tex_size,
                                    self.amplitude_exponent,
                                    self.n_textures)

        self.recalculate_stim() # and make sure there is something to draw even before we get going.
  


    def create_basic_textures(self, tex_size, amplitude_exponent, n_textures):
        t2 = int(tex_size/2)
        X,Y = np.meshgrid(np.linspace(-t2,t2,tex_size,endpoint=True),
                        np.linspace(-t2,t2,tex_size,endpoint=True))
        ecc = np.sqrt(X**2 + Y**2)
        ampl_spectrum = np.fft.fftshift(ecc**-amplitude_exponent, (0,1))
        phases = np.random.randn(n_textures, tex_size, tex_size) * 2 * np.pi
        textures = np.zeros((n_textures, tex_size, tex_size))
        # loop over different textures
        for nt in range(n_textures):
            compl_f = ampl_spectrum * np.sin(phases[nt]) + 1j * ampl_spectrum * np.cos(phases[nt])
            textures[nt] = np.fft.ifft2(compl_f).real
            # center at zero
            textures[nt] -= textures[nt].mean()
            # scale and clip to be within [-1,1]
            textures[nt] /= textures[nt].std()*6.666
            textures[nt] += 0.5
            textures[nt][textures[nt]<0] = 0
            textures[nt][textures[nt]>1] = 1

        return textures




    # evt. recalculate stim een mask meegeven die gelijk het mask is voor fixation 
    # functie ook een size meegeven die verschilt tussen fixation and surround (?)

    def recalculate_stim(self, red_boost=1, green_boost=1, blue_boost=0, opacity=None):
        which_textures = np.random.choice(self.n_textures, 3, replace=False)
        orientation = np.random.choice([0,90,180,270], 1)

        if opacity == None:
            opacity = self.opacity

        tex = np.zeros((int(self.tex_size), int(self.tex_size), 4))
        tex[:,:,0] = red_boost + self.basic_textures[which_textures[0]]
        tex[:,:,1] = green_boost + self.basic_textures[which_textures[1]]
        tex[:,:,2] = blue_boost + self.basic_textures[which_textures[2]]
        tex[:,:,3] = opacity * np.ones(self.basic_textures[0].shape)

        tex[tex>1] = 1

        self.noise_fest_stim = visual.GratingStim(self.session.screen, 
                                tex=tex, 
                                size=(self.size, self.size),
                                ori=orientation)



        ################################################
        # Create mask for each stimulus object
        ################################################

        self.stimulus_aperture = filters.makeMask(matrixSize=1080,    # TEMP SIZE PARAM?
                                shape='raisedCosine', 
                                radius=self.aperture,   
                                center=(0.0, 0.0),
                                range=[-1, 1], 
                                fringeWidth= 0.1)

        self.noise_fest_stim_new = visual.GratingStim(self.session.screen, 
                                        tex=tex, 
                                        mask=self.stimulus_aperture, 
                                        size=[self.size, self.size], 
                                        pos = np.array((0.0,0.0)),
                                        ori=orientation,
                                        color=(1.0, 1.0, 1.0))      
        ################################################



    def draw(self):
        #self.noise_fest_stim.draw()
        self.noise_fest_stim_new.draw()     # NEW


class PRFStim(object):  
    def __init__(self, session, 
                        cycles_in_bar=5, 
                        bar_width=1./8., 
                        tex_nr_pix=2048, 
                        flicker_frequency=6, 
                        max_eccentricity=512,
                        **kwargs):
        self.session = session
        self.cycles_in_bar = cycles_in_bar
        self.bar_width = bar_width
        self.tex_nr_pix = tex_nr_pix
        self.flicker_frequency = flicker_frequency
        self.max_eccentricity = max_eccentricity

        bar_width_in_pixels = self.tex_nr_pix*self.bar_width
        bar_width_in_radians = 2*np.pi*self.cycles_in_bar
        bar_pixels_per_radian = bar_width_in_radians/bar_width_in_pixels
        pixels_ls = np.linspace(-tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix)

        tex_x, tex_y = np.meshgrid(pixels_ls, pixels_ls)
        self.sqr_tex = np.sign(np.sin(tex_x) * np.sin(tex_y))

        self.sqr_tex[:,int(self.tex_nr_pix*(bar_width/2)+self.tex_nr_pix/2):] = 0
        self.sqr_tex[:,:int(-self.tex_nr_pix*(bar_width/2)+self.tex_nr_pix/2)] = 0


        self.checkerboard_1 = visual.GratingStim(self.session.screen, tex=self.sqr_tex, mask=None,
                          color=[1.0, 1.0, 1.0],
                          opacity=1.0,
                          size=(self.session.screen.size[1], self.session.screen.size[1]),
                          ori=0)
        self.checkerboard_2 = visual.GratingStim(self.session.screen, tex=-self.sqr_tex, mask=None,
                          color=[1.0, 1.0, 1.0],
                          opacity=1.0,
                          size=(self.session.screen.size[1], self.session.screen.size[1]),
                          ori=0)

    def draw(self, time, period, orientation, position_scaling):

        pos_in_ori = position_scaling * (-0.5 + int(0.5+time)/period) * self.max_eccentricity * 2

        x_pos, y_pos = cos((2.0*pi)*-orientation/360.0)*pos_in_ori, sin((2.0*pi)*-orientation/360.0)*pos_in_ori

        if np.sin(2*np.pi*time*self.flicker_frequency) > 0:
            self.checkerboard_1.setPos([x_pos, y_pos])
            self.checkerboard_1.setOri(orientation)
            self.checkerboard_1.draw()
        else:
            self.checkerboard_2.setPos([x_pos, y_pos])
            self.checkerboard_2.setOri(orientation)
            self.checkerboard_2.draw()























