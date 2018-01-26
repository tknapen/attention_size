from __future__ import division
from exptools.core.session import EyelinkSession
from trial import AttSizeTrial
from psychopy import visual, clock
import numpy as np
import sympy as sym
from math import * 
import os
import exptools
import json
import glob

from session import AttSizeSession


def ecc_deg_2_mm(eps, lmb=1.2, eps0=1.0):
    """Returns X, the 'eccentricity' in cm"""
    return lmb * np.log(1 + (eps/eps0))

def ecc_mm_2_deg(X, lmb=1.2, eps0=1.0):
    """Returns epsilon, the eccentricity in degrees"""
    return -eps0 + np.exp(X/lmb)


# Polarcoordinate function
def pol2cart(rho, phi):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return(x, y)


# Calculate arc of blob, related to circle radius
def line2arc(radius,distance):
     return 2*radius*math.asin(distance/(2*radius))


def PointsInCircum(r,n):
    return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in xrange(0,n+1)]


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
                row_spacing_factor=0.8, **kwargs):
        
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
        for ecc, bpr, s in zip(rows_ecc_in_deg[2:], nr_blobs_per_ring[1:], blob_row_blob_sizes[1:]):
            if ring_nr in ring_pos:
                continue
            ring_condition = floor(nr_rings * total_nr_rows / ring_nr)
            for pa in np.linspace(0, 2*np.pi,bpr):
                x, y = pol2cart(ecc, pa)
                element_array_np.append([x, y, ecc, pa, s, 0, 0, 0, ring_nr, ring_condition])
            ring_nr += 1
        self.element_array_np = np.array(element_array_np)

        self.element_array_stim = visual.ElementArrayStim(self.session.screen, 
                                                    colors=self.element_array_np[:,[5,6,7]], 
                                                    colorSpace='rgb', 
                                                    nElements=self.element_array_np.shape[0], 
                                                    sizes=self.element_array_np[:,4], 
                                                    sfs=0, 
                                                    xys=self.element_array_np[:,[0,1]])
        for i in range(nr_rings):
            self.repopulate_condition_ring_colors(condition_nr=i, color_balance=0.5)

        # rings are easy, blobs are hard
        self.rings = [psychopy.visual.Circle(self.session.screen, 
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

    def insert_parameters_into_stim(self):
        self.element_array_stim.setColors(self.element_array_np[:,[5,6,7]], log=False)
        self.element_array_stim.setXYs(self.element_array_np[:,[0,1]], log=False)

    def recalculate_elements(self):
        pass

    def draw(self):
        self.element_array_stim.draw()

class PRFStim(object):  
    def __init__(self, session, 
                        cycles_in_bar=5, 
                        bar_width=1./8., 
                        tex_nr_pix=2048, 
                        flicker_frequency=6, 
                        **kwargs):
        self.session = session
        self.cycles_in_bar = cycles_in_bar
        self.bar_width = bar_width
        self.tex_nr_pix = tex_nr_pix
        self.flicker_frequency = flicker_frequency

        bar_width_in_pixels = self.tex_nr_pix*self.bar_width
        bar_width_in_radians = 2*np.pi*self.cycles_in_bar
        bar_pixels_per_radian = bar_width_in_radians/bar_width_in_pixels
        pixels_ls = np.linspace(-tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix)

        tex_x, tex_y = np.meshgrid(pixels_ls, pixels_ls)
        self.sqr_tex = np.sign(np.sin(tex_x) * np.sin(tex_y))
        self.sqr_tex[:,self.tex_nr_pix*(bar_width/2)+self.tex_nr_pix/2:] = 0
        self.sqr_tex[:,:-self.tex_nr_pix*(bar_width/2)+self.tex_nr_pix/2] = 0

        #INITIALISE SOME STIMULI
        self.checkerboard = visual.GratingStim(self.session.screen, tex=self.sqr_tex, mask=None,
                          color=[1.0, 1.0, 1.0],
                          opacity=1.0,
                          size=(self.tex_nr_pix, self.tex_nr_pix),
                          ori=0)

    def draw(time=0, period, orientation, position):
        checkerboard_polarity = np.sign(np.sin(2*np.pi*time*self.flicker_frequency))

        self.checkerboard.setTex(self.sqr_tex*checkerboard_polarity)
        self.checkerboard.setPos([-self.tex_nr_pix/2 + int(0.5+time)*self.tex_nr_pix/period, 0])
        self.checkerboard.setOri(orientation)

        self.checkerboard.draw()























