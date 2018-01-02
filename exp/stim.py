from exptools.core.session import EyelinkSession
from trial import AttSizeTrial
from psychopy import visual, clock
import numpy as np
import sympy as sym
import os
import exptools
import json
import glob

from session import AttSizeSession


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
    def __init__(self, nr_rings=3, ecc_min=0.1, ecc_max=5, nr_blob_rows_per_ring=24, row_spacing_factor=0.8, session, **kwargs):
        
        total_nr_rows = nr_rings*(nr_blob_rows_per_ring+1)+1

        self.mm_cortex_from_fovea, self.ecc_range, self.cm_trace = mm_cortex_from_fovea(ecc_min=ecc_min, ecc_max=ecc_max, nr=total_nr_rows)
        # Ring positions
        ring_pos = np.arange(0, total_nr_rows, nr_blob_rows_per_ring+1)
        blob_row_pos = np.setdiff1d(np.arange(total_nr_rows), ring_pos)
        blob_row_sizes = np.diff(ecc_range)
        blob_row_blob_sizes = blob_row_sizes * row_spacing_factor

        nr_blobs_per_ring = ((2*np.pi*ecc_range[1:])/blob_row_sizes).astype(int)

        xys = [[pol2cart(ecc, pa) for pa in np.linspace(0, 2*np.pi,bpr)+np.random.rand()] for ecc, bpr in zip(ecc_range[2:], nr_blobs_per_ring[1:])]
        pl.figure()
        for i, xy in enumerate(xys):
            pl.plot(np.array(xy)[:,0], np.array(xy)[:,1], 'ko', ms=100*blob_row_blob_sizes[i])

        # rings are easy, blobs are hard
        self.rings = [psychopy.visual.Circle(self.session.screen, 
                                            radius=self.ecc_range[rp]*self.session.pixels_per_degree, 
                                            lineColor=[-1, -1, -1], 
                                            edges=256) for rp in ring_pos]



        ring_bins = np.digitize(self.mm_cortex_from_fovea, np.linspace(ecc_min,self.mm_cortex_from_fovea.max(), nr_rings, endpoint=False))
        max_ring_positions_deg = np.arange(len(ring_bins))[np.r_[np.diff(ring_bins)!=0, True]]
        max_ring_positions_pix = max_ring_positions_deg * session.pixels_per_degree

        self.ringposition_degrees, self.ringposition_pixels = self.degreestopixels_rings(nr_rings)


        ########################################################################################################################
        # Binning over different eccentricities per X and mm
        # Recalculating pixels from the degrees for blob amounts, locations and sizes
        ########################################################################################################################

        total_blob_rows = (rows_of_blobs * conditions_rings)+3
        blob_bins = np.digitize(mm_cortex_from_fovea, np.linspace(ecc_min,mm_cortex_from_fovea.max(), total_blob_rows, endpoint=False))
        max_blob_positions = ecc_range[np.r_[np.diff(ring_bins)!=0, True]]

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

        blobposition_degrees, blobposition_pixels = degreestopixels_blobs(max_blob_positions)

    def degreestopixels_rings(conditions_rings):
        ringposition_degrees = []
        ringposition_pixels = []

        for x in range (0,len(max_ring_positions)):
            deg = max_ring_positions[x,0]
            ringposition_degrees.append(deg)
            pix = (deg*10) * 10.63074
            ringposition_pixels.append(pix)

        return ringposition_degrees, ringposition_pixels


    def recalculate_elements(self):
        pass

    def draw(self):
        pass

class PRFStim(object):  
    __init__(self, **kwargs):
        pass