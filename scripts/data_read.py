
####################################################################################################
# Read in data from attention experiment - JR 2018
####################################################################################################

'''
- Add gender + age + task variables to participant info

- Double check psychometric curves with:
    - psychometric_curve_interval: 0.15,
    - psychometric_curve_nr_steps_oneside: 6,

-> compare datafiles from course to own and rearrange
-> make into pandas file with titles and stuff

'''

####################################################################################################
# Import modules/libraries
####################################################################################################

from __future__ import division

import gc, numpy
from psychopy import data, visual, logging, core, event

import matplotlib
matplotlib.use('Qt4Agg')  # change this to control the plotting 'back end'
import matplotlib.pyplot as plt

import pylab
import pickle

import numpy as np
from scipy.optimize import curve_fit


####################################################################################################
# Custom functions
####################################################################################################

def zeroDivide(num, den, replace=0):
	"""
	Divide num by den, ignoring zero divide errors (will replace with replace value)
	"""
	# Do divide, be quiet about errors
	with np.errstate(divide='ignore', invalid='ignore'):
		res = num / den
	# zero divide errors may produce nans or infs, so replace any of these
	res[np.isnan(res) | np.isinf(res)] = replace
	# return
	return res

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y


def percentagecorrect(a, b):
    a = float(a)
    b = float(b)
    correctperc = a / b * 100
    correctperc = round(correctperc,2)
    return correctperc


####################################################################################################
# Read in datafile
####################################################################################################

file = '../exp/data/RJ_1_2018-04-21_11.28.24_outputDict.pkl'
subj = file[-39:-37]

with open(file, 'r') as f:
	d = pickle.load(f)
    
params = d['parameterArray']
other = d['eventArray']

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
bg_values = [d[''u'bg_trial_stimulus_value'] for d in params]			# bg values  (neg is red, pos is green)
responses = [d[''u'answer'] for d in params]							# responses: 0 green, 1 red   
trials = len(responses)

# Change to NP arrays
fix_values= np.array(fix_values)
bg_values = np.array(bg_values)
responses = np.array(responses)


# ####################################################################################################
# # Missed trials 
# ####################################################################################################

raw_data_array = [fix_values[responses!=-1], bg_values[responses!=-1], responses[responses!=-1]]
nw_trials = (len(raw_data_array[1]))
#print(raw_data_array)

print("> Removed %s trials from original of %i trials" % (trials-nw_trials, trials))

# ####################################################################################################
# # Performance within both conditions 
# ####################################################################################################

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

# Percentage correct bg condition
fixation_condition_pc = percentagecorrect(sum(fix_condition),len(fix_condition))     
background_condition_pc = percentagecorrect(sum(bg_condition),len(bg_condition))     

print('> Percentage correct fixation condtion: %s percent' % (fixation_condition_pc)) 
print('> Percentage correct background condtion: %s percent' % (background_condition_pc)) 

n_fixation_condition = np.array(fix_condition)
n_background_condition = np.array(bg_condition)

# Make new data array (with filtered data) with all the data combined
n_fix_values = raw_data_array[0][:]
n_bg_values = raw_data_array[1][:]
n_responses = raw_data_array[2][:]

data_array = [n_fix_values, n_bg_values, n_responses, n_fixation_condition, n_background_condition]
# (data_array[0][:]) = fix val column
# (data_array[1][:]) = bg val column
# (data_array[2][:]) = resp val column
# (data_array[3][:]) = fix correct responses column         
# (data_array[4][:]) = bg correct responses column         
#print(data_array)


# ####################################################################################################
# # Fit psychometric curve
# ####################################################################################################

# with open('./data.pkl', 'w') as fd:
# 	pickle.dump({'xdata':xdata, 'ydata':ydata}, fd)

# allIntensities: negative intensity = red / positive intensity green
# allResponses: 0 = green / red = 1	

# proportion red selection (expect 50% but might be shifted)
# for both conditions a psychometric curve
# -> response 'red' vs actual presence of red (point of subjective equality)
# -> intensity values on x axis / fraction of red response on y axis

# another 2 curves for performance: if red becomes more visible, higher change of correct,
# but only for the active condition, the other one should be chance level



####################################################################################################
# 1 Fixation condition color balance
####################################################################################################

# fix_color = data_array[0][:]	# fix values			
# ydata1 = data_array[2][:]		# responses (1 red, 0 green)		

# fix_color_xCounts, fix_color_xBins = np.histogram(fix_color, bins=10) 				# get prop "red" resps per bin / # get bins and counts overall (ignore resp type) for x
# fix_color_xBinMids = fix_color_xBins[:-1] + np.diff(fix_color_xBins)/2						# get midpoints of x bins (used for plotting)
# fix_color_redCounts = np.histogram(fix_color[ydata1==1], bins=fix_color_xBins)[0]	# get counts for just "red" resps
# fixpropRedCounts = zeroDivide(fix_color_redCounts, fix_color_xCounts)				# normalise redCounts by xCounts to get propRedCounts

# popt, pcov = curve_fit(sigmoid, fix_color, ydata1)
# # print (popt)

# x = np.linspace(fix_color.min(), fix_color.max(), 100)
# y = sigmoid(x, *popt)

# pylab.plot(fix_color_xBinMids, fixpropRedCounts, 'o', label='prop red counts')
# pylab.plot(x,y, label='fit')
# pylab.axhline(y=0.5, linestyle='dashed')
# pylab.xlabel('All red - all green')
# pylab.ylabel('Percentage Red responses')
# pylab.legend(loc='best')
# pylab.title('Color Balance: Fixation condition')
# pylab.show()
# #plt.close()


####################################################################################################
# 2 Surround condition color balance
####################################################################################################

# bg_color = data_array[1][:]		# bg values	
# ydata2 = data_array[2][:]		# responses (1 red, 0 green)	

# xCounts, xBins = np.histogram(bg_color, bins=10) 				# get prop "red" resps per bin / # get bins and counts overall (ignore resp type) for x
# xBinMids = xBins[:-1] + np.diff(xBins)/2						# get midpoints of x bins (used for plotting)
# redCounts = np.histogram(bg_color[ydata2==1], bins=xBins)[0]	# get counts for just "red" resps
# fixpropRedCounts = zeroDivide(redCounts, xCounts)				# normalise redCounts by xCounts to get propRedCounts

# popt, pcov = curve_fit(sigmoid, bg_color, ydata2)
# # print (popt)

# x = np.linspace(bg_color.min(), bg_color.max(), 100)
# y = sigmoid(x, *popt)

# pylab.plot(xBinMids, fixpropRedCounts, 'o', label='prop red counts')
# pylab.plot(x,y, label='fit')
# pylab.axhline(y=0.5, linestyle='dashed')	
# pylab.xlabel('All red - all green')
# pylab.ylabel('Percentage Red responses')
# pylab.legend(loc='best')
# pylab.title('Color Balance: Surround condition')
# pylab.show()
# plt.close()



####################################################################################################
# 3 Fixation condition performance
####################################################################################################

# fx_color = data_array[0][:]                         # fix values     
# ydata3 = data_array[3][:]                           # fix correct responses column

# # Select red value trials only 
# fix_red_values_only = [fx_color[fx_color <= 0], ydata3[fx_color <= 0]]  # select red trials only
# fix_red_color_values = fix_red_values_only[0][:]
# ydata3 = fix_red_values_only[1][:]

# # Rescale for proportion red pixels (only red included) (-15 is 100% red, 0 is 0% red)
# fix_stim_red_proportion = ((abs(fix_red_color_values))/0.15)*100       # make absolute and percent of total            

# xCounts, xBins = np.histogram(fix_stim_red_proportion, bins=5)              # get prop "red" resps per bin / # get bins and counts overall (ignore resp type) for x
# xBinMids = xBins[:-1] + np.diff(xBins)/2                                # get midpoints of x bins (used for plotting)
# redCounts = np.histogram(fix_stim_red_proportion[ydata3==1], bins=xBins)[0] # get counts for just "red" resps
# fixpropRedCounts = zeroDivide(redCounts, xCounts)                       # normalise redCounts by xCounts to get propRedCounts

# popt, pcov = curve_fit(sigmoid, fix_stim_red_proportion, ydata3)
# # print (popt)

# x = np.linspace(fix_stim_red_proportion.min(), fix_stim_red_proportion.max(), 100)
# y = sigmoid(x, *popt)

# pylab.plot(xBinMids, fixpropRedCounts, 'o', label='prop correct')
# pylab.plot(x,y, label='fit')
# pylab.axhline(y=0.5, linestyle='dashed')
# pylab.xlabel('0 percent red - 100 percent red')
# pylab.ylabel('Percentage Correct')
# pylab.legend(loc='best')
# pylab.title('Performance: Fixation condition')
# pylab.show()
# plt.close()


####################################################################################################
# Fixation + Surround condition color
####################################################################################################

# performance or color??

fig, subfigs = pylab.subplots(1,2, sharey=True)

condition = 0       # fixation condition
x_values = np.sort(np.unique(data_array[condition]))    # fixation values
mean_responses = np.array([data_array[2][data_array[condition]==x].mean() for x in x_values]) # responses

popt, pcov = curve_fit(sigmoid, x_values, mean_responses)

x = np.linspace(x_values.min(), x_values.max(), 100)
y = sigmoid(x, *popt)

subfigs[condition].plot(x_values, mean_responses, 'o', label='data')
subfigs[condition].plot(x,y, label='fit')
subfigs[condition].axhline(y=0.5, linestyle='dashed')
subfigs[condition].set_ylim([-0.05,1.05])
subfigs[condition].set_xlabel('Stim color (all red, 0, all green)')
subfigs[condition].set_ylabel('% awnsered red')
subfigs[condition].legend(loc='best')
subfigs[condition].set_title('Fixation Condition (pp: %s)' %subj)

condition = 1       # surround condition
x_values = np.sort(np.unique(data_array[condition]))
mean_responses = np.array([data_array[2][data_array[condition]==x].mean() for x in x_values])

popt, pcov = curve_fit(sigmoid, x_values, mean_responses)
#print (popt)

x = np.linspace(x_values.min(), x_values.max(), 100)
y = sigmoid(x, *popt)

subfigs[condition].plot(x_values, mean_responses, 'o', label='data')
subfigs[condition].plot(x,y, label='fit')
subfigs[condition].axhline(y=0.5, linestyle='dashed')
subfigs[condition].set_ylim([-0.05,1.05])
subfigs[condition].set_xlabel('Stim color (all red, 0, all green)')
subfigs[condition].set_ylabel('% awnsered red')
subfigs[condition].legend(loc='best')
subfigs[condition].set_title('Surround Condition (pp: %s)' %subj)
pylab.show()

#subfigs.close()


# shell()

# find x and y values corresponding in graph
# xvalues = fig4[0].get_xdata()
# yvalues = fig4[0].get_ydata()

# idx = np.where(xvalues == 50) 
# y = yvalues[idx[0][0]]



