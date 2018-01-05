

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







from psychopy import visual, core, event

#create a window to draw in
myWin = visual.Window((1200,1200), allowGUI=False, units='pix')

tex_nr_pix=2048
bar_width = 1/8.0
cycles_in_bar = 5

bar_width_in_pixels = tex_nr_pix*bar_width
bar_width_in_radians = 2*np.pi*cycles_in_bar
bar_pixels_per_radian = bar_width_in_radians/bar_width_in_pixels
pixels_ls = np.linspace(-tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix/2*bar_pixels_per_radian,tex_nr_pix)

tex_x, tex_y = np.meshgrid(pixels_ls, pixels_ls)
sqr_tex = np.sign(np.sin(tex_x) * np.sin(tex_y))
sqr_tex[:,tex_nr_pix*(bar_width/2)+tex_nr_pix/2:] = 0
sqr_tex[:,:-tex_nr_pix*(bar_width/2)+tex_nr_pix/2] = 0

#INITIALISE SOME STIMULI
grating1 = visual.GratingStim(myWin, tex=sqr_tex, mask=None,
                  color=[1.0, 1.0, 1.0],
                  opacity=1.0,
                  size=(tex_nr_pix, tex_nr_pix),
                  ori=0)

directions_per_time = np.concatenate([np.r_[np.ones(60+np.random.randint(60)), -np.ones(60+np.random.randint(60))] for x in range(200)])


trialClock = core.Clock()
t = 0
TR = 1.5
frame_nr = 0

while t < 20*TR:    # quits after 20 secs

    t = trialClock.getTime()

    # sqr_tex = np.roll(sqr_tex, int(6*directions_per_time[frame_nr]), axis=0)

    if frame_nr%5:
        sqr_tex = -sqr_tex
        # print((-tex_nr_pix/2 + int(0.5+t/TR)*tex_nr_pix/20.0, directions_per_time[frame_nr]))
    grating1.setTex(sqr_tex)  # drift at 1Hz
    grating1.draw()  #redraw it

    grating1.setPos([-tex_nr_pix/2 + int(0.5+t/TR)*tex_nr_pix/20.0, 0])
    grating1.setOri(0)

    myWin.flip()          #update the screen

    #handle key presses each frame
    for keys in event.getKeys():
        if keys in ['escape','q']:
            core.quit()

    frame_nr += 1


