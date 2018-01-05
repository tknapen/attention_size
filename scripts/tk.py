

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

%pylab

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

nr_rings=3
ecc_min=0.1
ecc_max=5
nr_blob_rows_per_ring=8
row_spacing_factor=0.8




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

xys = [[pol2cart(ecc, pa) for pa in np.linspace(0, 2*np.pi,bpr)+np.random.rand()] for ecc, bpr in zip(rows_ecc_in_deg[2:], nr_blobs_per_ring[1:])]
figure()
for i, xy in enumerate(xys):
    plot(np.array(xy)[:,0], np.array(xy)[:,1], 'ko', ms=100*blob_row_blob_sizes[i])