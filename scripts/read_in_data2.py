
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
filenames_tsv = glob.glob(newpath + '*.tsv')

#### Which run corresponds to which condition #### 
condition_1 = filenames_tsv[0]#, filenames_tsv[2] 			# look at fixation circle
condition_2 = filenames_tsv[1]# , filenames_tsv[3] 			# look at background circle


#### + add different contrast options as well!!!!!

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


# Combine data from each condition
# data_c1 = (pd.read_csv(f, sep='\t') for f in condition_1)
# all_data_c1   = pd.concat(data_c1, ignore_index=True)

# data_c2 = (pd.read_csv(f, sep='\t') for f in condition_2)
# all_data_c2   = pd.concat(data_c2, ignore_index=True)

condition_1 = pd.DataFrame([all_data_c1['fix_trial_stimulus_value'], all_data_c1['bg_trial_stimulus_value'], all_data_c1['answer']]).T
condition_1_data = condition_1[condition_1['answer']!=-1]

condition_2 = pd.DataFrame([all_data_c2['fix_trial_stimulus_value'], all_data_c2['bg_trial_stimulus_value'], all_data_c2['answer']]).T
condition_2_data = condition_2[condition_2['answer']!=-1]


########################################################################################################################################
# General params from data
########################################################################################################################################

# Original total trials
trials_c1 = (len(condition_1))
trials_c2 = (len(condition_2))


# Count missed trials condition 1
general_ans_c1 = condition_1['answer'].value_counts()
missed_trials_c1 = general_ans_c1[-1]

# Count missed trials condition 2
general_ans_c2 = condition_2['answer'].value_counts()
missed_trials_c2 = general_ans_c2[-1]


# Condition 1 Percentage ans green and red (- missed trials)
fil_total_trials_c1 = trials_c1 - missed_trials_c1
green_proportion_c1 = general_ans_c1[0]/fil_total_trials_c1
red_proportion_c1 = general_ans_c1[1]/fil_total_trials_c1

# Condition 2 Percentage ans green and red (- missed trials)
fil_total_trials_c2 = trials_c2 - missed_trials_c2
green_proportion_c2 = general_ans_c2[0]/fil_total_trials_c2
red_proportion_c2 = general_ans_c2[1]/fil_total_trials_c2





#### Kan alleen als alle contrasten in de file hetzelfde zijn !!!! ###
# Condition 1 Contrast intensity 
fixation_contrast = float(all_data_c1.loc[0]['pixstim_fix_opacity'])

# Condition 2 Contrast intensity 
background_contrast = float(all_data_c2.loc[0]['pixstim_bg_opacity'])


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
# Plot curves
########################################################################################################################################

# Loop over all conditions and contrasts, and make figure for all





########################################################################################################################################
# Fixation psych curve
########################################################################################################################################

condition = 'fix_trial_stimulus_value'
fix = pd.DataFrame([condition_1_data[condition], condition_1_data['answer']]).T

fix = pd.DataFrame([condition_1_data['fix_trial_stimulus_value'], condition_1_data['answer']]).T
fix = fix[fix['answer']!=-1]
fix_gr = fix.groupby('fix_trial_stimulus_value')

fix_x_values = fix['fix_trial_stimulus_value'].unique()
fix_x_values.sort()
fix_y_values = np.array(fix_gr.mean()).ravel()

popt, pcov = curve_fit(sigmoid, fix_x_values, fix_y_values)

fix_x_continuous = np.linspace(fix_x_values.min(), fix_x_values.max(), 1000)
fix_y_continuous = sigmoid(fix_x_continuous, *popt)


# Find the x value for y value 0.5
y_stuff = np.round(fix_y_continuous, 3)
cord = np.where(y_stuff == 0.5)

# Catch error if finding x breaks
try: # if y actually crosses 0, give x coordinate
	fix_x_value = fix_y_continuous[cord]			# array of x values
	fix_x_value = round(fix_x_value[0], 4,) 		# changed to optimize saving in image, takes first value

except: # if not, it is empty
	fix_x_value = str('never')


# Plot figure
con1 = pl.figure()
pl.plot(fix_x_values, fix_y_values, 'o', label='data')
pl.plot(fix_x_continuous, fix_y_continuous, label='fit')
pl.axhline(y=0.5, linestyle='dashed')
pl.gca().set_ylim([-0.05,1.05])
pl.gca().set_xlabel('Stim color (all red, 0, all green) | Crosses at x = %s' %(fix_x_value))
pl.gca().set_ylabel('% awnsered red')
pl.gca().legend(loc='best')
pl.gca().set_title('Performance for Fixation Circle \n From %s | Contrast intensity: %s' %(subj, fixation_contrast))
pl.show()



# ##############################

# # # Try to save figure in file
# contrast = str(fixation_contrast)
# con1.savefig(newpath + condition + contrast + '.pdf', bbox_inches='tight')

##############################



# ########################################################################################################################################
# # Background psych curve
# ########################################################################################################################################

condition = 'bg_trial_stimulus_value'

bg = pd.DataFrame([condition_1_data['bg_trial_stimulus_value'], condition_1_data['answer']]).T
bg = bg[bg['answer']!=-1]
bg_gr = bg.groupby('bg_trial_stimulus_value')

bg_x_values = bg['bg_trial_stimulus_value'].unique()
bg_x_values.sort()
bg_y_values = np.array(bg_gr.mean()).ravel()

popt, pcov = curve_fit(sigmoid, bg_x_values, bg_y_values)

bg_x_continuous = np.linspace(bg_x_values.min(), bg_x_values.max(), 1000)
bg_y_continuous = sigmoid(bg_x_continuous, *popt)


# Find the x value for y value 0.5
y_stuff = np.round(bg_y_continuous, 3)
cord = np.where(y_stuff == 0.5)

# Catch error if finding x breaks
try: # if y actually crosses 0, give x coordinate
	bg_x_value = bg_x_continuous[cord]			# array of x values
	bg_x_value = round(bg_x_value[0], 4,) 		# changed to optimize saving in image, takes first value


except: # if not, it is empty
	bg_x_value = str('never')


# Plot figure
pl.plot(bg_x_values, bg_y_values, 'o', label='data')
pl.plot(bg_x_continuous, bg_y_continuous, label='fit')
pl.axhline(y=0.5, linestyle='dashed')
pl.gca().set_ylim([-0.05,1.05])
pl.gca().set_xlabel('Stim color (all red, 0, all green) | Crosses at x = %s' %(bg_x_value))
pl.gca().set_ylabel('% awnsered red')
pl.gca().legend(loc='best')
pl.gca().set_title('Performance for Background Circle \n From %s | Contrast intensity: %s' %(subj, background_contrast))
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
print("  - Contrast intensity: "+"{:.2%}".format(fixation_contrast));
print("  - Task percentage correct: "+"{:.2%}".format(condition1_pc_correct));
print("> - At y = 0.5, x = %s" % (fix_x_value))

# Print stuff for condition 2
print("######################################################################")
print(" > Condition 2 (background) from %s \n  - Total runs: %i trials" % (subj, trials_c2))
print("  - Removed %s missed trials, now %i trials" % (missed_trials_c2, len(condition_2_data)))
print("  - Proportion answered green: "+"{:.2%}".format(green_proportion_c2))
print("  - Proportion answered red: "+"{:.2%}".format(red_proportion_c2))
print("  - Contrast intensity: "+"{:.2%}".format(background_contrast))
print("  - Task percentage correct: "+"{:.2%}".format(condition2_pc_correct));
print("> - At y = 0.5 at x = %s" % (bg_x_value))
print("######################################################################")







