
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

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y


########################################################################################################################################
# Change these general params
########################################################################################################################################

#### For which subject? ####
subj = 'sub02/'

newpath = '/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/' + subj
filenames = glob.glob(newpath + '*.tsv')

### Which run corresponds to which condition #### 
condition_1 = filenames[0]#, filenames[2] 			# look at fixation circle
condition_2 = filenames[1]# , filenames[3] 			# look at background circle

### Change when using multiple contrasts ####

########################################################################################################################################
# Read in & combine all data per condition
########################################################################################################################################

# Read in for condition 1
if len(condition_1) > 50:	# if only 1 file, len will be amount of trials
	all_data_c1 = (pd.read_csv(condition_1, sep='\t')) 

elif len(condition_1) < 50: # if more files, len will be amount of files
	data_c1 = (pd.read_csv(f, sep='\t') for f in condition_1)
	all_data_c1   = pd.concat(data_c1, ignore_index=True)

else: print('Something crashed during read-in csv files')


# Read in for condition 2
if len(condition_2) > 50:	# if only 1 file, len will be amount of trials
	all_data_c2 = (pd.read_csv(condition_2, sep='\t')) 

elif len(condition_2) < 50: # if more files, len will be amount of files
	data_c2 = (pd.read_csv(f, sep='\t') for f in condition_2)
	all_data_c2   = pd.concat(data_c2, ignore_index=True)

else: print('Something crashed during read-in csv files')


########################################################################################################################################
# General params from data
########################################################################################################################################

# Throw out all missed trials
condition_1_data = all_data_c1[all_data_c1['answer']!=-1]
condition_2_data = all_data_c2[all_data_c2['answer']!=-1]

# Original total trials
trials_c1 = (len(all_data_c1))
trials_c2 = (len(all_data_c1))

# Count missed trials condition 1
general_ans_c1 = all_data_c1['answer'].value_counts()
missed_trials_c1 = general_ans_c1[-1]

# Count missed trials condition 2
general_ans_c2 = all_data_c2['answer'].value_counts()
missed_trials_c2 = general_ans_c2[-1]


# Condition 1 Percentage ans green and red (- missed trials)
fil_total_trials_c1 = trials_c1 - missed_trials_c1
green_proportion_c1 = general_ans_c1[0]/fil_total_trials_c1
red_proportion_c1 = general_ans_c1[1]/fil_total_trials_c1

# Condition 2 Percentage ans green and red (- missed trials)
fil_total_trials_c2 = trials_c2 - missed_trials_c2
green_proportion_c2 = general_ans_c2[0]/fil_total_trials_c2
red_proportion_c2 = general_ans_c2[1]/fil_total_trials_c2



########################################################################################################################################
# Correct or incorrect responses per condition
########################################################################################################################################

# (-values are red, + values green. Response 0 green, 1 for red)

# Condition 1 
condition_1_data['fix_performance'] = (condition_1_data['fix_trial_stimulus_value'] < 0.0) & (condition_1_data['answer'] == 1.0) | (condition_1_data['fix_trial_stimulus_value'] > 0.0) & (condition_1_data['answer'] == 0.0)
condition_1_data['fix_performance'] = condition_1_data['fix_performance'] *1  # *1 makes true values 1, and false 0
condition1_pc_correct = np.sum(condition_1_data['fix_performance']) / len(condition_1_data)

# Condition 2
condition_2_data['bg_performance'] = (condition_2_data['bg_trial_stimulus_value'] < 0.0) & (condition_2_data['answer'] == 1.0) | (condition_2_data['bg_trial_stimulus_value'] > 0.0) & (condition_2_data['answer'] == 0.0) 
condition_2_data['bg_performance'] = condition_2_data['bg_performance'] *1  # *1 makes true values 1, and false 0
condition2_pc_correct = np.sum(condition_2_data['bg_performance']) / len(condition_2_data)


########################################################################################################################################
# Make figures for datasets
########################################################################################################################################

datasets = condition_1_data, condition_2_data
stimuli = 'fix_trial_stimulus_value', 'bg_trial_stimulus_value'

for x in datasets:

	for s in stimuli:

		stim = s
		data = pd.DataFrame([x[s], x['answer']]).T
		data_gr = data.groupby(s)

		x_values = data[s].unique()
		x_values.sort()
		y_values = np.array(data_gr.mean()).ravel()

		popt, pcov = curve_fit(sigmoid, x_values, y_values)

		x_continuous = np.linspace(x_values.min(), x_values.max(), 1000)
		y_continuous = sigmoid(x_continuous, *popt)

		############################################################################
		# Find the x value for y value 0.5
		############################################################################

		y_stuff = np.round(y_continuous, 3)
		cord = np.where(np.logical_and(y_stuff>=0.49, y_stuff<=0.51))

		# Catch error if finding x breaks
		try: # if y actually crosses 0, give x coordinate
			x_value = y_continuous[cord]			# array of x values
			x_value = round(x_value[0], 4,) 		# changed to optimize saving in image, takes first value

		except: # if not, it is empty
			x_value = str('never')

		############################################################################
		# Get correct contrast intensity
		############################################################################
		if stim == 'fix_trial_stimulus_value':
			contrast = x.loc[1]['pixstim_fix_opacity']

		elif stim == 'bg_trial_stimulus_value':
			contrast = x.loc[1]['pixstim_bg_opacity']
		else:
			print('Cannot find contrast intensity in data')

		############################################################################
		# Plot figure
		############################################################################
		fig = pl.figure()
		pl.plot(x_values, y_values, 'o') #, label='data')
		pl.plot(x_continuous, y_continuous) # label='fit')
		pl.axhline(y=0.5, linestyle='dashed')
		pl.gca().set_ylim([-0.05,1.05])
		pl.gca().set_xlabel('Stim color (all red, 0, all green) | Crosses at x = %s' %(x_value))
		pl.gca().set_ylabel('% awnsered red')
		pl.gca().legend(loc='best')
		pl.gca().set_title('Performance for %s \n From %s | Contrast intensity: %s' %(s, subj, contrast))
		pl.show()



########################################################################################################################################
# Print out stuff
########################################################################################################################################

# Print stuff for condition 1
print("\n######################################################################")
print(" > Condition 1 (fixation) from %s \n  - Total runs: %i trials" % (subj, trials_c1))
print("  - Removed %s missed trials, now %i trials" % (missed_trials_c1, len(condition_1_data)))
print("  - Proportion answered green: "+"{:.2%}".format(green_proportion_c1))
print("  - Proportion answered red: "+"{:.2%}".format(red_proportion_c1));
print("  - Overal percentage correct: "+"{:.2%}".format(condition1_pc_correct));

# Print stuff for condition 2
print("######################################################################")
print(" > Condition 2 (background) from %s \n  - Total runs: %i trials" % (subj, trials_c2))
print("  - Removed %s missed trials, now %i trials" % (missed_trials_c2, len(condition_2_data)))
print("  - Proportion answered green: "+"{:.2%}".format(green_proportion_c2))
print("  - Proportion answered red: "+"{:.2%}".format(red_proportion_c2))
print("  - Overal percentage correct: "+"{:.2%}".format(condition2_pc_correct));
print("######################################################################")







