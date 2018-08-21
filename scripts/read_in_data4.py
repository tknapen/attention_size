
########################################################################################################################################
# Read in data in panda files - JR 2018
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


########################################################################################################################################
# General params
########################################################################################################################################

#### For which subject? ####
subj = 'sub02/'

newpath = '/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/' + subj
filenames = glob.glob(newpath + '*.tsv')

### Which run corresponds to which condition #### 
c1_contr1 = filenames[0]#, filenames[2]
c1_contr2 = filenames[1]#, filenames[3]
c2_contr1 = filenames[2]
c2_contr2 = filenames[3]

### Update these according to stuff above #### 
datasets = c1_contr1, c1_contr2, c2_contr1, c2_contr2
datanames = 'c1_contr1', 'c1_contr2', 'c2_contr1', 'c2_contr2'

print('\n###################################################################')
print('Dataset for %s' %(subj))
print('###################################################################')

########################################################################################################################################
# Read in & combine all data 
########################################################################################################################################

counter = 1

data_big = []

for x in datasets:
	data_name = 'Data' + str(counter)

	if len(x) > 50:		# if only 1 file, len will be amount of trials
		x = (pd.read_csv(x, sep='\t'))
		org_len = len(x)					# check original amount of trials
		x = x[x['answer']!=-1]				# remove all missed trials
		vars()[data_name] = x               # save dataset under data something etc per run
		data_big.append(vars()[data_name])	# append this dataset to large list of all sets

		misses = org_len - len(x)					# check new amount of trials
		print('> %s: Removed %s misses of original %s trials' %(data_name, misses, org_len))
		counter += 1

	elif len(x) < 50:	# if more files, len will be amount of files
		data = (pd.read_csv(f, sep='\t') for f in x)
		all_data = pd.concat(data, ignore_index=True)	# combine the sets
		org_len = len(all_data)							# check original amount of trials
		all_data = all_data[all_data['answer']!=-1]		# remove all missed trials
		vars()[data_name] = all_data
		data_big.append(vars()[data_name])

		misses = org_len - len(all_data)				# check new amount of trials
		print('> %s: Removed %s misses of original %s trials' %(data_name, misses, org_len))
		counter += 1

	else: print('Something crashed during read-in csv files')

print('###################################################################')



########################################################################################################################################
# Correct or incorrect responses per condition
########################################################################################################################################
# (-values are red, + values green. Response 0 green, 1 for red)

counter = 1

for d in data_big:
	data_name = 'Data' + str(counter)	# to rename the sets again

	fix = 'fix_performance'
	d[fix] = (d['fix_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['fix_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[fix] = d[fix] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d[fix]) / len(d))*100),2)

	contrast = d.iloc[0]['pixstim_fix_opacity']
	print('> %s: %s = %s percent for contrast %s' %(data_name, fix, pc_corr, contrast))


	bg = 'bg_performance'
	d[bg] = (d['bg_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['fix_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[bg] = d[bg] *1  # *1 makes true values 1, and false 0
	pc_corr = round(((np.sum(d[bg]) / len(d))*100),2)
	
	contrast = d.iloc[0]['pixstim_bg_opacity']
	print('> %s: %s = %s percent for contrast %s' %(data_name, bg, pc_corr, contrast))

	vars()[data_name] = d
	counter += 1

print('###################################################################')


########################################################################################################################################
# Make figures for datasets
########################################################################################################################################

stimuli = 'fix_trial_stimulus_value', 'bg_trial_stimulus_value'

figures = 1

for x in data_big:

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
		pl.show()


		############################################################################
		# Save figure
		############################################################################

		#Save figure in file
		fig.savefig(newpath + str(figures) + '.png', bbox_inches='tight')
		figures += 1
