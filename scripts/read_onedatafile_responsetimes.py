
########################################################################################################################################
# Read in data in files - JR 2018
########################################################################################################################################

from __future__ import division
import glob		
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'  # turns off overwrite warning

import pickle
import re

import scipy as sp
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use('Qt4Agg')  # change this to control the plotting 'back end'
import matplotlib.pyplot as plt

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

def percentagecorrect(a, b):
    a = float(a)
    b = float(b)
    correctperc = a / b * 100
    correctperc = round(correctperc,2)
    return correctperc




########################################################################################################################################
# Read in one datafile pickle.        change filename here
########################################################################################################################################

# Read in pickle file

###
file = '/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/sub02/sub02_4_2018-07-19_13.32.57_outputDict.pkl'
###

subj = file[-39:-37]

with open(file, 'r') as f:
	d = pickle.load(f)
    
params = d['parameterArray']
other = d['eventArray']


# Extract phase times from eventarray
all_runs = []

for runs in other:
	all_times_in_run = []

	for t in runs:
		time = re.findall('\d*\.?\d+',t)		# extract all numbers per run
		time = time[-1]							# select last numbers (the time)
		all_times_in_run.append(float(time))	# append all times from run 

	all_runs.append(all_times_in_run)			# above but for all runs

del all_runs[0]			# remove first trial


########################################################################################################################################
# All runs, times only, extract trial durations
########################################################################################################################################

all_runs_starttimes = []

for tm in all_runs:
	start = tm[0]
	all_runs_starttimes.append(start)

trial_durations = [x-y for x, y in zip(all_runs_starttimes[1:], all_runs_starttimes)]    

plt.plot(trial_durations)
plt.xlabel('Trials')
plt.ylabel('Time in seconds')
plt.ylim((1.15, 1.35))
plt.title('All trial durations')
plt.show()


########################################################################################################################################
# All runs, extract time of event (keypress)
########################################################################################################################################

all_runs_keypress = []

for tm in all_runs:
	start = tm[0]
	event = tm[-1]
	time = event - start
	all_runs_keypress.append(time)


plt.plot(all_runs_keypress)
plt.xlabel('Trials')
plt.ylabel('Time')
plt.title('All trial durations')
plt.show()


trials = len(trial_durations)
x = np.arange(trials)


plt.bar(x, trial_durations, color='blue')
plt.plot(all_runs_keypress, 'o', color = 'orange')
plt.xlabel('Trials')
plt.ylabel('Time')
plt.title('All trial durations')
plt.show()


########################################################################################################################################
# Take other values
########################################################################################################################################


# Get general params
gen_params = params[0]

in_scanner = gen_params[u'in_scanner']                   
if in_scanner == 0:
    in_scanner = 'outside'
else:
    in_scanner = 'inside'

# Stimulus selection
stim_type = gen_params[u'which_stimulus_type']                   
if stim_type == 0:
    stim_type = 'blobs'
    fix_opacity = gen_params[u'blobstim_fix_opacity']
    bg_opacity = gen_params[u'blobstim_bg_opacity']

else:
    stim_type = 'pixels'
    fix_opacity = gen_params[u'pixstim_bg_opacity']
    bg_opacity = gen_params[u'pixstim_fix_opacity']


print("______________________________________________________________")
print("> Participant file of %s, %s the scanner" % (subj, in_scanner))
print("> Experiment version was '%s', fix opacity %s and bg opacity %s" % (stim_type, fix_opacity, bg_opacity))


# Get relevant params                                         
fix_values = [d[''u'fix_trial_stimulus_value'] for d in params]			# fix values (neg is red, pos is green)
del fix_values[0]			# remove first trial

bg_values = [d[''u'bg_trial_stimulus_value'] for d in params]			# bg values  (neg is red, pos is green)
del bg_values[0]			# remove first trial

responses = [d[''u'answer'] for d in params]							# responses: 0 green, 1 red 
del responses[0]			# remove first trial


trials = len(responses)


# Change to NP arrays
fix_values= np.array(fix_values)
bg_values = np.array(bg_values)
responses = np.array(responses)
responsetimes = np.array(all_runs_keypress)

####################################################################################################
# Missed trials 
####################################################################################################

raw_data_array = [fix_values[responses!=-1], bg_values[responses!=-1], responses[responses!=-1], responsetimes[responses!=-1]]
nw_trials = (len(raw_data_array[1]))
#print(raw_data_array)

print("> Removed %s trials from original of %i trials" % (trials-nw_trials, trials))





####################################################################################################
# Correlate responsetimes with task difficulty
####################################################################################################

# x = abs(raw_data_array[1])
# y = raw_data_array[3]

# plt.scatter(x,y)
# plt.xlabel('Fixation Values')
# plt.ylabel('Responsetimes')
# plt.show()





####################################################################################################
# Performance within both conditions 
####################################################################################################

# Condition outcome for fixation and surround
# (raw_data_array[0][:]) = fix val column
# (raw_data_array[1][:]) = bg val column
# (raw_data_array[2][:]) = resp val column

fix_condition = []
for x in range(0,nw_trials):
    fix_val = raw_data_array[0][x]
    resp = int(raw_data_array[2][x])
        
    if fix_val < 0.00 and resp == 1:       # if negative number (red) and resp is red
        fix_condition.append(1)            # correct 
        
    elif fix_val > 0.00 and resp == 0:     # if positive number (green) and resp is green
        fix_condition.append(1)            # correct     
    
    else:
        fix_condition.append(0) 


bg_condition = []
for x in range(0,nw_trials):
    bg_val = raw_data_array[1][x]
    resp = int(raw_data_array[2][x])
        
    if bg_val < 0.00 and resp == 1:       # if negative number (red) and resp is red
        bg_condition.append(1)            # correct 
        
    elif bg_val > 0.00 and resp == 0:     # if positive number (green) and resp is green
        bg_condition.append(1)            # correct     
    
    else:
        bg_condition.append(0) 




####################################################################################################

# x = fix_condition
# y = raw_data_array[3]

# plt.scatter(x,y)
# plt.xlabel('Correct or incorrect (0,1')
# plt.ylabel('Responsetimes')
# plt.show()






