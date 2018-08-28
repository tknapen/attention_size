
########################################################################################################################################
# Read in data in panda files - JR 2018
# For each condition a figure
########################################################################################################################################

# Current errors
# - 

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



def remove_misses(x):
	data = (pd.read_csv(f, sep='\t') for f in x)
	org_len = len(x)					# check original amount of trials
	all_data = pd.concat(data, ignore_index=True)	# combine the sets
	org_len = len(all_data)							# check original amount of trials
	all_data = all_data[all_data['answer']!=-1]		# remove all missed trials

	misses = org_len - len(all_data)					# check new amount of trials
	miss_info = str(data_name) + ' removed ' + str(misses) + ' from original ' + str(org_len)	# save missed info in string
	print('> Removed %s misses of original %s trials' %(misses, org_len))	# print missed info
	
	return all_data




def performance(d, val):
# (- values are red, + values green. Response 0 green, 1 for red)
	d['performance'] = (d[val] < 0.0) & (d['answer'] == 1.0) | (d[val] > 0.0) & (d['answer'] == 0.0)
	d['performance'] = d['performance'] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d['performance']) / len(d))*100),2)





# hieronder moet nog	

	contrast = d.iloc[0]['pixstim_fix_opacity']
	print('> Condition 1: %s = %s percent for contrast %s' %(fix, pc_corr, contrast))
	vars()[data_name] = c1



bla = performance(c1, 'fix_trial_stimulus_value')



########################################################################################################################################
# General params
########################################################################################################################################

#### For which subject? ####
subj = 'sub03/'

newpath = '/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/' + subj
filenames = glob.glob(newpath + '*.tsv')

### Which run corresponds to which condition #### 
condition1_fixation = filenames[0], filenames[2], filenames[4]
condition2_background = filenames[1], filenames[3], filenames[5]

### Update these according to stuff above #### 
datasets = condition1_fixation, condition2_background


print('\n###################################################################')
print('Dataset for %s' %(subj))
print('###################################################################')


########################################################################################################################################
# Read in & combine all data 
########################################################################################################################################

c1 = remove_misses(condition1_fixation)
c2 = remove_misses(condition2_background)




########################################################################################################################################
# Correct or incorrect responses per condition
########################################################################################################################################
# (- values are red, + values green. Response 0 green, 1 for red)


for d in data_big:
	data_name = 'Data' + str(counter)	# to rename the sets again

	fix = 'fix_performance'
	c1[fix] = (d['fix_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['fix_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[fix] = d[fix] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d[fix]) / len(d))*100),2)

	contrast = d.iloc[0]['pixstim_fix_opacity']
	print('> Condition 1: %s = %s percent for contrast %s' %(fix, pc_corr, contrast))
	vars()[data_name] = c1
	counter += 1






for d in data_big:
	data_name = 'Data' + str(counter)	# to rename the sets again

	# Condition 1
	fix = 'fix_performance'
	d[fix] = (d['fix_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['fix_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[fix] = d[fix] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d[fix]) / len(d))*100),2)

	contrast = d.iloc[0]['pixstim_fix_opacity']
	print('> Condition 1: %s = %s percent for contrast %s' %(fix, pc_corr, contrast))

	perf_info = str(data_name) + ' for ' + str(fix) + ' percent correct ' + str(pc_corr)# + ' for contrast ' + str(contrast)	# save performance info in string
	performance_info.append(perf_info)


	# Condition 2
	bg = 'bg_performance'
	d[bg] = (d['bg_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['bg_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[bg] = d[bg] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d[bg]) / len(d))*100),2)
	
	contrast = d.iloc[0]['pixstim_bg_opacity']
	print('> Condition 2: %s = %s percent for contrast %s' %(bg, pc_corr, contrast))

	perf_info = str(data_name) + ' for ' + str(bg) + ' percent correct ' + str(pc_corr)# + ' for contrast ' + str(contrast)	# save performance info in string
	performance_info.append(perf_info)


	vars()[data_name] = d
	counter += 1

print('###################################################################')


########################################################################################################################################
# Make figures for datasets
########################################################################################################################################

figures = 1
stimuli = 'fix_trial_stimulus_value', 'bg_trial_stimulus_value'



for x in data_big:

	for s in stimuli:

		stim = s
		data = pd.DataFrame([x[s], x['answer']]).T
		data_gr = data.groupby(s)

		x_values = data[s].unique()
		x_values.sort()
		y_values = np.array(data_gr.mean()).ravel()


		# Fitting non-linear least squares fit
		try: 
			# Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds.
			popt, pcov = curve_fit(sigmoid, x_values, y_values, method='trf') 

		except:
			# Standard scipy fit procedure
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
			contrast = x.iloc[1]['pixstim_fix_opacity']

		elif stim == 'bg_trial_stimulus_value':
			contrast = x.iloc[1]['pixstim_bg_opacity']
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
		#pl.show()


		############################################################################
		# Save figures in png file
		############################################################################

		fig.savefig(newpath + str(figures) + '_figure' + '.png', bbox_inches='tight')
		figures += 1
		

############################################################################
# Save txt file with specs
############################################################################

filename = '_info_' + '.txt'			# make textfile with same rel figure name
outF = open(newpath + filename, "w")
outF.write('####################################################################' + '\n')
outF.write('Dataset for ' + str(subj)+ '\n')
outF.write('####################################################################' + '\n')
for miss in missed_info:
	outF.write('> '+ str(miss)+ '\n')
outF.write('####################################################################' + '\n')
for per in performance_info:
	outF.write('> '+ str(per)+ '\n')
outF.write('####################################################################' + '\n')	
outF.close()




