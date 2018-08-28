
########################################################################################################################################
# Read in data in panda files - JR 2018
########################################################################################################################################

from __future__ import division
import glob		
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'  # turns off overwrite warning

import scipy as sp
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use('Qt4Agg')  # change this to control the plotting 'back end'
import matplotlib.pyplot as pl

# def sigmoid(x, x0, k):
#      y = 1 / (1 + np.exp(-k*(x-x0)))
#      return y


# def func(x, x0, k):
#      y = 1 / (1 + np.exp(-k*(x-x0)))
#      return y

def func(x, a, b, c):
	return a * np.exp(-b * x) + c

#def func(x, alpha, beta):
#    return 1. / (1 + np.exp( -(x-alpha)/beta ))

########################################################################################################################################
# Read in one datafile (OLD)
########################################################################################################################################

# f = pd.read_csv('/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/sub02/sub02_2_2018-07-19_13.20.10.tsv', sep='\t')
f = pd.read_csv('/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/sub03/sub03_4_2018-08-21_14.31.33.tsv', sep='\t')

# Show all variables in data
print(f.iloc[0])

# Give info on all variables
f.info()

# Create dataframe with relevant values (T at the end for flipping)
data = pd.DataFrame([f['fix_trial_stimulus_value'], f['bg_trial_stimulus_value'], f['answer'], f['bar_orientation']]).T

# Remove entire row from answer if -1 / missed response
filtered_data = data[data['answer']!=-1]

# Print entire index from one param
print(filtered_data.loc[:,'answer']) 

print("\n######################################################################")

# Original total trials
trials = (len(f))
print("> Total runs: %i trials" % (trials))

# Count missed trials
general_ans = f['answer'].value_counts()
missed_trials = general_ans[-1]
print("> Removed %s missed trials, now %i trials" % (missed_trials, len(f)-missed_trials))

# Percentage ans green and red (minus the missed trials)
fil_total_trials = len(f) - missed_trials

green_proportion = general_ans[0]/fil_total_trials
print("> Proportion answered green: "+"{:.2%}".format(green_proportion));

red_proportion = general_ans[1]/fil_total_trials
print("> Proportion answered green: "+"{:.2%}".format(red_proportion));

# Contrast intensity of circles
fixation_contrast = float(f.loc[0]['pixstim_fix_opacity']*100)
background_contrast = float(f.loc[0]['pixstim_bg_opacity']*100)
print("> Fixation contrast %s \n> Background contrast %s" % (fixation_contrast, background_contrast))


########################################################################################################################################
# Correct or incorrect responses
########################################################################################################################################

# (-values are red, + values green. Response 0 green, 1 for red)

# Fixation circle performance
filtered_data['fix_performance'] = (filtered_data['fix_trial_stimulus_value'] < 0.0) & (filtered_data['answer'] == 1.0) | (filtered_data['fix_trial_stimulus_value'] > 0.0) & (filtered_data['answer'] == 0.0) 
filtered_data['fix_performance'] = filtered_data['fix_performance'] *1  # *1 makes true values 1, and false 0

# Background circle performance
filtered_data['bg_performance'] = (filtered_data['bg_trial_stimulus_value'] < 0.0) & (filtered_data['answer'] == 1.0) | (filtered_data['bg_trial_stimulus_value'] > 0.0) & (filtered_data['answer'] == 0.0) 
filtered_data['bg_performance'] = filtered_data['bg_performance'] *1  # *1 makes true values 1, and false 0

# Percentage correct for circles 
fix_pc_correct = np.sum(filtered_data['fix_performance']) / len(filtered_data)
bg_pc_correct = np.sum(filtered_data['bg_performance']) / len(filtered_data)

print("> Percentage correct fixation stimulus: "+"{:.2%}".format(fix_pc_correct));
print("> Percentage correct background stimulus: "+"{:.2%}".format(bg_pc_correct));



#######################################################################################################################################
# Select for each bar orientation, the amount of correct trials for the relevant task
########################################################################################################################################

## Which condition is relevant? ##
condition = 'fix_trial_stimulus_value'


# geeft alleen nans terug.. 
# filtered_data.loc[filtered_data['fix_trial_stimulus_value'] == 1, "bar_orientation_correct_trial"] = filtered_data["bar_orientation"]
# print(filtered_data['bar_orientation_correct_trial'])

# # If performance is 1 (thus correct), output that bar orientation in variable 'bar_orientation_correct_trial'
# filtered_data.loc[filtered_data[condition] == 1, "bar_orientation_correct_trial"] = filtered_data["bar_orientation"]
# print(filtered_data['bar_orientation_correct_trial'])

# # Count unique values in 'bar_orientation_correct_trial' and output in 'bar_orientation_unique'
# filtered_data['bar_orientation_unique'] = filtered_data.groupby(['bar_orientation_correct_trial'])[condition].transform('count')

# bar_influence= filtered_data['bar_orientation_correct_trial']
# bar_influence = bar_influence[~np.isnan(bar_influence)]
# gr = bar_influence.value_counts()

# # Determine unqiue variables in orientations + remove the nans 
# x = np.unique(filtered_data['bar_orientation_correct_trial'])
# x = x[~np.isnan(x)]

# pl.hist(bar_influence[~np.isnan(bar_influence)], x)
# pl.title('Bar Orientations for correct ans from %s' %condition )
# pl.xlabel("Bar Orientation")
# pl.ylabel("Total Correct Responses")
# fig = pl.gcf()
# pl.show()




########################################################################################################################################
# Fixation psych curve
########################################################################################################################################

fix = pd.DataFrame([filtered_data['fix_trial_stimulus_value'], filtered_data['answer']]).T
fix = fix[fix['answer']!=-1]

fix_gr = fix.groupby('fix_trial_stimulus_value')
fix_x_values = fix['fix_trial_stimulus_value'].unique()
fix_x_values.sort()
fix_y_values = np.array(fix_gr.mean()).ravel()

popt, pcov = curve_fit(func, fix_x_values, fix_y_values)#, method='trf')

fix_x_continuous = np.linspace(fix_x_values.min(), fix_x_values.max(), 1000)
fix_y_continuous = func(fix_x_continuous, *popt)

pl.plot(fix_x_values, fix_y_values, 'o', label='data')
pl.plot(fix_x_continuous, fix_y_continuous, label='fit')
pl.axhline(y=0.5, linestyle='dashed')
pl.gca().set_ylim([-0.05,1.05])
pl.gca().set_xlabel('Stim color (all red, 0, all green)')
pl.gca().set_ylabel('% awnsered red')
pl.gca().legend(loc='best')
pl.gca().set_title('Performance for Fixation Circle')
pl.show()


# Find the x value for y value 0.5
y_stuff = np.round(fix_y_continuous, 2)
cord = np.where(y_stuff == 0.5)

# Catch error if finding x breaks
try: # if y actually crosses 0, give x coordinate
	x_value = x_continuous[cord]			# array of x values
	x_value = round(x_value[0], 4,)			# changed to optimize saving in image, takes first value

except: # if not, it is empty
	x_value = str('never')


print("> Fixation: Chance level y = 0.5 at x= %s" % (x_value))


########################################################################################################################################
# Background psych curve
########################################################################################################################################

bg = pd.DataFrame([filtered_data['bg_trial_stimulus_value'], filtered_data['answer']]).T
bg = bg[bg['answer']!=-1]

bg_gr = bg.groupby('bg_trial_stimulus_value')				# creates table with means for each unique value
bg_gr.describe()											# open table

bg_x_values = bg['bg_trial_stimulus_value'].unique()		# store unique x values
bg_x_values.sort()											# sort them
bg_y_values = np.array(bg_gr.mean()).ravel()

## Proposed new
#m = bg.groupby(['bg_trial_stimulus_value']).mean() 


popt, pcov = curve_fit(func, bg_x_values, bg_y_values)#, method='trf')

bg_x_continuous = np.linspace(bg_x_values.min(), bg_x_values.max(), 1000)
bg_y_continuous = func(bg_x_continuous, *popt)

pl.plot(bg_x_values, bg_y_values, 'o', label='data')
pl.plot(bg_x_continuous, bg_y_continuous, label='fit')
pl.axhline(y=0.5, linestyle='dashed')
pl.gca().set_ylim([-0.05,1.05])
pl.gca().set_xlabel('Stim color (all red, 0, all green)')
pl.gca().set_ylabel('% awnsered red')
pl.gca().legend(loc='best')
pl.gca().set_title('Performance for Background Circle')
pl.show()


# Find the x value for y value 0.5
y_stuff = np.round(bg_y_continuous, 3)
cord = np.where(y_stuff == 0.5)
bg_x_value = bg_x_continuous[cord]

print("> Background: Chance level y = 0.5 at x= %s" % (bg_x_value))





