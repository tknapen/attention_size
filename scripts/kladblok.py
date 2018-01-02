
########################################################################################################################
# Python Experiment - Jutta de Jong
# Updated 9 December 2017
########################################################################################################################

# Current random blob selection should be staircased
#   --> Van te voren indeelbaar maken, de verhoudingen tussen elkaar, voor psychophysics
# output is een psychometric curve als check: y probability responding with green, x percentage green blobs
# dan wil je voor de relevante conditie 75% correct, en de andere 2 50% correct


# Bepaal colortask kleuren, niet meer door aantal blobs, maar oppervlakte blobs!
# blobs > oppervlakte > blobs
# instructie: meer rood of groen oppervlak? bv 4 kleine, 4 grote, groot weegt meer mee) (diameter per blob dus ipv aantal)
# dus niet zozeer dots..

# Issues:
# 1 Blob amount has two solutions, which one is better? Blob size is based on space between each row that is drawn.
#   Amount of blobs on circle scales based on row width, but only lines out well if amount is * 2, for both solutions??
#   Also outer rows sometimes have less blobs than the ones before, weird
# 2 Drawn ring positions (i.e. condition rings) are fixed. So blob rows in the widest ring are positioned wrong,
#   so outer ring is further away due to the magnification factor that is used (it's correct but looks weird)

# To-do: blob location should be in between rings, not on rings, plus offset of first blob random(0-360)
# To-do: Check if the min-eccentricity should be 0.1 or 0.5. For now I skipp the first circle size because too small

########################################################################################################################

condition = 1
trials = 50
rows_of_blobs = 2

ecc_min = 0.1                                   # Minimum Eccentricity (should be 0.1 for scanner)
ecc_max = 3.75                                  # Maximum Eccentricity (should be 0.5 for scanner, now 3/4)

screenwidth = 797.25                            # should be 1063.0 for scanner
screenheight = 797.25                           # should be 1063.0 for scanner

TR = 1.5
time_per_trial = TR/2                           # 2 trials per TR?
conditions_rings = 3                            # Always 3 rings present


########################################################################################################################
# Import Libraries
########################################################################################################################

import sys
sys.ps1 = 'SOMETHING'       # Fixes an interactive backend problem on my mac

import psychopy
import random
import numpy as np
import sympy as sp
import time
import copy
from psychopy import visual, core, event
import math
import os
from random import randint
from math import pi

win = visual.Window([(screenheight), (screenwidth)], units='pix', monitor='testMonitor')

########################################################################################################################
# Participant Info
########################################################################################################################

import psychopy.gui
gui = psychopy.gui.Dlg()

gui.addField("Participant:")
gui.addField("Condition:")
gui.addField("Run:")

gui.show()

subj_id = gui.data[0]
cond_num = gui.data[1]
condition = int(cond_num)
run = gui.data[2]

data_path = "subj0" +subj_id + "_run" + run + "_con" + cond_num + ".tsv"

if os.path.exists(data_path):
    sys.exit("Data path " + data_path + " already exists!")


########################################################################################################################
# Required functions
########################################################################################################################

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
#  Fixation Cross
########################################################################################################################

fixx = psychopy.visual.Line(win=win, units="pix", lineColor=[-1, -1, -1], lineWidth=2)
fixx.start = [0, -10]
fixx.end = [(0), 10]

fixy = psychopy.visual.Line(win=win, units="pix", lineColor=[-1, -1, -1], lineWidth=2)
fixy.start = [-10, 0]
fixy.end = [(10), 0]


########################################################################################################################
# Cortical Magnification factor & Eccentricities
########################################################################################################################

def cm(ecc, mm_0, tau):
    return mm_0 * np.exp(-ecc*tau)

mm0 = sp.Symbol('mm0')
ecc = sp.Symbol('ecc')
tau = sp.Symbol('tau')
sp.integrate(sp.exp(-ecc*tau), ecc)

ecc_nr = 20000
ecc_range = np.linspace(ecc_min,ecc_max,ecc_nr)
mm_0 = 12
tau = 0.5
fs = 14

cm_trace = cm(ecc_range, mm_0, tau)
mm_cortex_from_fovea = np.cumsum(cm_trace) / (ecc_nr/ecc_max)


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

ringposition_degrees, ringposition_pixels = degreestopixels_rings(conditions_rings)


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

blobposition_degrees, blobposition_pixels = degreestopixels_blobs(max_blob_positions)


########################################################################################################################
# Determine which rows the blobs will be present + the blobs size
########################################################################################################################

# Remove first + last bins, plus the rows you won't show
max_nr = (len(blobposition_pixels))
cut_blobposition_pixels = blobposition_pixels[1:max_nr-1]              # Cut first and last var from total positions
final_blobposition_pixels = cut_blobposition_pixels[:]                 # Because python copies variables correctly
cut1 = final_blobposition_pixels[rows_of_blobs]
cut2 = final_blobposition_pixels[rows_of_blobs+(rows_of_blobs+1)]      # Cut 2 vars that won't show blobs, depend on total rows
final_blobposition_pixels.remove(cut1)
final_blobposition_pixels.remove(cut2)


# Determine blob size (from blob position, blob size = (blob row + 1) - blob row)
for x in range (0, len(blobposition_pixels)):
    filled_blob_sizes = []

    for y in range (0, len(blobposition_pixels)-1):
        size = blobposition_pixels[y+1] - blobposition_pixels[y]
        filled_blob_sizes.append(size)


# Cut out the blob sizes, from the blob rows you won't show
final_blobsize  = copy.deepcopy(filled_blob_sizes)
cutb1 = final_blobsize[rows_of_blobs]
cutb2 = final_blobsize[rows_of_blobs + (rows_of_blobs + 1)]
cutb3 = final_blobsize[rows_of_blobs + (rows_of_blobs + 1) + (rows_of_blobs + 1)]  # Cut 2 vars, depend on total rows

final_blobsize.remove(cutb1)
final_blobsize.remove(cutb2)
final_blobsize.remove(cutb3)



########################################################################################################################
# Determine Blob amount for each blob row
########################################################################################################################

x_blobs_per_row = []
blob_locations = []

# Aantal blobs = diameter ring / size blobs * (spacing factor)
for x in range(0,len(final_blobsize)):
    r = final_blobposition_pixels[x]
    diam_for_ring = 2*r
    x_blobs = diam_for_ring / final_blobsize[x] * 3     # ???
    x_blobs_per_row.append(int(x_blobs))


########################################################################################################################
# Determine which conditions based on blob amounts + rows per condition
########################################################################################################################

condition_1_blobs = x_blobs_per_row[0:0+rows_of_blobs]
condition_2_blobs = x_blobs_per_row[0+rows_of_blobs:0+rows_of_blobs+rows_of_blobs]
condition_3_blobs = x_blobs_per_row[0+rows_of_blobs+rows_of_blobs:0+rows_of_blobs+rows_of_blobs+rows_of_blobs]

condition_1 = sum(condition_1_blobs)
condition_2 = sum(condition_2_blobs)
condition_3 = sum(condition_3_blobs)
total_blobs = condition_1+condition_2+condition_3

########################################################################################################################
# Determine Blob locations for each blob row
########################################################################################################################


for x in range (0,len(final_blobsize)):
    r = final_blobsize[x]
    n = x_blobs_per_row[x]
    locations = PointsInCircum(r,n)
    blob_locations.append(locations)


blobParameterList = []
blobcoords = []
blobsize = []


for ringI in range(0, len(blob_locations)):
    blob_params = []

    blobRho = final_blobposition_pixels[ringI]
    amount_per_row = x_blobs_per_row[ringI]
    currentBlobSize = final_blobsize[ringI]

    PhiPerBlob = 2 * math.pi/amount_per_row
    HalfPhi = PhiPerBlob/2
    oddRing = ringI%2

    for blobI in range(0, amount_per_row):
        blob_params.append([blobRho, (PhiPerBlob * blobI) + (HalfPhi * oddRing), currentBlobSize])
        blobsize.append(currentBlobSize)
    blobParameterList.append(blob_params)


# Change params to polar-coordinates
for blobList in blobParameterList:
    for blob in blobList:
        blobcoords.append(pol2cart(blob[0], blob[1]))


########################################################################################################################
# Determine blob colors
########################################################################################################################

def random_colors():
    colormatrix_con1 = []
    colormatrix_con2 = []
    colormatrix_con3 = []
    red_condition_1 = []
    green_condition_1 = []
    red_condition_2 = []
    green_condition_2 = []
    red_condition_3 = []
    green_condition_3 = []

    for c in range (0,1):

        condition_1t = float(copy.deepcopy(condition_1))
        condition_2t = float(copy.deepcopy(condition_2))
        condition_3t = float(copy.deepcopy(condition_3))

        ####################################################################################
        # Condition 1 Selection
        ####################################################################################

        #selection_criterium_condition_1 = np.linspace(20, 80, 20)
        #select = random.choice(selection_criterium_condition_1)

        # Determine which color is dominant
        for colors in range(0, 1):
            leadcolor_c1 = random.choice([(255, 0, 0), (0, 255, 0)])

            if leadcolor_c1 == (255, 0, 0):     # Red
                othercolor_c1 = (0, 255, 0)
            elif leadcolor_c1 == (0, 255, 0):   # Green
                othercolor_c1 = (255, 0, 0)

        # Pick random nummer, determine percentage it is from total blobs this condition
        manipulated_blobs_condition_1 = float(randint(1, condition_1))

        for r in range(0, int(manipulated_blobs_condition_1)):
            colormatrix_con1.append(leadcolor_c1)

        for g in range(0, int(condition_1t - manipulated_blobs_condition_1)):
            colormatrix_con1.append(othercolor_c1)

        random.shuffle(colormatrix_con1, random.random)


        ####################################################################################
        # Condition 2 Selection
        ####################################################################################

        # selection_criterium_condition_2 = np.linspace(20, 80, 20)
        # select = random.choice(selection_criterium_condition_2)

        # Determine which color is dominant
        for colors in range(0, 1):
            leadcolor_c2 = random.choice([(255, 0, 0), (0, 255, 0)])

            if leadcolor_c2 == (255, 0, 0):     # Red
                othercolor_c2 = (0, 255, 0)
            elif leadcolor_c2 == (0, 255, 0):   # Green
                othercolor_c2 = (255, 0, 0)

        # Pick random nummer, determine percentage it is from total blobs this condition
        select2 = float(randint(1, condition_2))
        manipulated_blobs_condition_2 = round(select2)

        for r in range(0, int(manipulated_blobs_condition_2)):
            colormatrix_con2.append(leadcolor_c2)

        for g in range(0, int(condition_2 - manipulated_blobs_condition_2)):
            colormatrix_con2.append(othercolor_c2)

        random.shuffle(colormatrix_con2, random.random)


        ####################################################################################
        # Condition 3 Selection
        ####################################################################################

        # selection_criterium_condition_3 = np.linspace(20, 80, 20)
        # select = random.choice(selection_criterium_condition_3)

        # Determine which color is dominant
        for colors in range(0, 1):
            leadcolor_c3 = random.choice([(255, 0, 0), (0, 255, 0)])

            if leadcolor_c3 == (255, 0, 0):     # Red
                othercolor_c3 = (0, 255, 0)
            elif leadcolor_c3 == (0, 255, 0):   # Green
                othercolor_c3 = (255, 0, 0)

        # Pick random nummer, determine percentage it is from total blobs this condition
        select3 = float(randint(1, condition_3))
        manipulated_blobs_condition_3 = round(select3)

        for r in range(0, int(manipulated_blobs_condition_3)):
            colormatrix_con3.append(leadcolor_c3)

        for g in range(0, int(condition_3 - manipulated_blobs_condition_3)):
            colormatrix_con3.append(othercolor_c3)

        random.shuffle(colormatrix_con3, random.random)


        ####################################################################################
        # Or hardcode all manipulations
        ####################################################################################

        # Make less hardcoded, more like 10% around or something??
        # if condition == 1:
        #     c1_1 = 60
        #     c1_2 = 70
        #     c1_3 = 80
        #
        #     perc_difference_condition_1 = round((condition_1t/100) * c1_1)
        #     perc_difference_condition_2 = round((condition_2t/100) * c1_2)
        #     perc_difference_condition_3 = round((condition_3t/100) * c1_3)
        #
        # elif condition ==2:
        #     c2_1 = 50
        #     c2_2 = 60
        #     c2_3 = 80
        #
        #     perc_difference_condition_1 = round((condition_1t / 100) * c2_1)
        #     perc_difference_condition_2 = round((condition_2t / 100) * c2_2)
        #     perc_difference_condition_3 = round((condition_3t / 100) * c2_3)
        #
        #
        # else:
        #     c3_1 = 50
        #     c3_2 = 60
        #     c3_3 = 80
        #
        #     perc_difference_condition_1 = round((condition_1t / 100) * c2_3)
        #     perc_difference_condition_2 = round((condition_2t / 100) * c2_2)
        #     perc_difference_condition_3 = round((condition_3t / 100) * c2_3)

        # ++++ color selection etc etc


        ####################################################################################

        # Count color per condition
        red_condition_1 = colormatrix_con1.count((255, 0, 0))
        green_condition_1 = colormatrix_con1.count((0, 255, 0))
        red_condition_2 = colormatrix_con2.count((255, 0, 0))
        green_condition_2 = colormatrix_con2.count((0, 255, 0))
        red_condition_3 = colormatrix_con3.count((255, 0, 0))
        green_condition_3 = colormatrix_con3.count((0, 255, 0))

    return colormatrix_con1, colormatrix_con2, colormatrix_con3, red_condition_1, green_condition_1, \
           red_condition_2, green_condition_2, red_condition_3, green_condition_3


########################################################################################################################
# Draw the rings and the blobs
########################################################################################################################

def runexperiment(trials):
    all_trial_params = []
    roundz = 0

    while roundz < trials:
        start_trial = round(time.time(),4)          # Start time trial
        fixx.draw(), fixy.draw()                    # Draw fixation cross

        # Draw rings
        for ringI in range(0, 4):
                radius = ringposition_pixels[ringI]
                ring = psychopy.visual.Circle(win=win, units="pix", radius=radius, lineColor=[-1, -1, -1], edges=128)
                ring.draw()

        # Get colormatrices for each condition
        colormatrix_con1, colormatrix_con2, colormatrix_con3, red_condition_1, green_condition_1, \
        red_condition_2, green_condition_2, red_condition_3, green_condition_3 = random_colors()

        # Combine to one matrix with colors
        colormatrix = colormatrix_con1 + colormatrix_con2 + colormatrix_con3

        # Draw all the blobs
        all_blobs = visual.ElementArrayStim(win, colors=colormatrix, colorSpace='rgb', nElements=total_blobs, sizes=blobsize, sfs=0, xys=blobcoords)
        all_blobs.draw()

        win.flip()
        core.wait(time_per_trial)

        # Determine keypress and time
        keys = psychopy.event.getKeys( keyList=["left", "right"], timeStamped=psychopy.core.Clock())

        for key in keys:
            if key[0] == "left":
                key_num = 90
            else:
                key_num = 30

        # No response = 404
        if keys:
            response = key_num
            responsetime = key[1]
        else: # if keys does not exist
            response = 404
            responsetime = 404

        roundz += 1
        end_trial = time.time()
        trial_duration = round(end_trial - start_trial,4)

        trial_params = [roundz, trial_duration, response, responsetime, red_condition_1, green_condition_1,
           red_condition_2, green_condition_2, red_condition_3, green_condition_3]
        all_trial_params.append(trial_params)

    return all_trial_params


########################################################################################################################
# Check and save data
########################################################################################################################

all_trial_params = runexperiment(trials)
np.savetxt(data_path, all_trial_params, fmt='%1.2f')
