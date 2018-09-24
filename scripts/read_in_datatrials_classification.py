
########################################################################################################################################
# Read in data in panda files - JR 2018
# For all files figures
########################################################################################################################################

# Current errors
# - Now get the y = 0.5ish value for x... but should take the one of the curve instead

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
import matplotlib.pyplot as plt

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

# def sigmoid(x, a, b, c):
# 	return a * np.exp(-b * x) + c


########################################################################################################################################
# General params
########################################################################################################################################

#### For which subject? ####		<----
subj = 'sub02/'

newpath = '/Users/jong/Dropbox/Spinoza_Centre_for_Neuroimaging/Project_Selective_Attention/attention_size/exp/data/' + subj
filenames = glob.glob(newpath + '*.tsv')

### Which run corresponds to which run #### 			<----
c1_contr1 = filenames[0]
c2_contr1 = filenames[1]

c1_contr2 = filenames[2]
c2_contr2 = filenames[3]

# c1_contr3 = filenames[4]
# c2_contr3 = filenames[5]

# c1_contr4 = filenames[6]
# c2_contr4 = filenames[7]

# c1_contr5 = filenames[8]
# c2_contr5 = filenames[9]

### Update these according to stuff above #### 									<----
datasets = c1_contr1, c2_contr1, c1_contr2, c2_contr2#, c1_contr3, c2_contr3, c1_contr4, c2_contr4, c1_contr5, c2_contr5


print('\n###################################################################')
print('Dataset for %s' %(subj))
print('###################################################################')


########################################################################################################################################
# Read in & combine all data 
########################################################################################################################################

counter = 1
data_big = []
missed_info = []

for x in datasets:
	data_name = 'Data' + str(counter)

	if len(x) > 50:		# if only 1 file, len will be amount of trials
		x = (pd.read_csv(x, sep='\t'))
		org_len = len(x)					# check original amount of trials
		x = x[x['answer']!=-1]				# remove all missed trials
		vars()[data_name] = x               # save dataset under data something etc per run
		data_big.append(vars()[data_name])	# append this dataset to large list of all sets

		misses = org_len - len(x)					# check new amount of trials
		miss_info = str(data_name) + ' removed ' + str(misses) + ' from original ' + str(org_len)	# save missed info in string
		print('> %s: Removed %s misses of original %s trials' %(data_name, misses, org_len))	# print missed info
		missed_info.append(miss_info)															# append saved missed stuff

		counter += 1

	elif len(x) < 50:	# if more files, len will be amount of files
		data = (pd.read_csv(f, sep='\t') for f in x)
		all_data = pd.concat(data, ignore_index=True)	# combine the sets
		org_len = len(all_data)							# check original amount of trials
		all_data = all_data[all_data['answer']!=-1]		# remove all missed trials
		vars()[data_name] = all_data
		data_big.append(vars()[data_name])

		misses = org_len - len(all_data)				# check new amount of trials
		miss_info = str(data_name) + ' removed ' + str(misses) + ' from original ' + str(org_len)	# save missed info in string
		print('> %s: Removed %s misses of original %s trials' %(data_name, misses, org_len))	# print missed info
		missed_info.append(miss_info)															# append saved missed stuff

		counter += 1

	else: print('Something crashed during read-in csv files')

print('###################################################################')



########################################################################################################################################
# Correct or incorrect responses per condition
########################################################################################################################################
# (- values are red, + values green. Response 0 green, 1 for red)

counter = 1
performance_info = []
performances = []
contrasts = []

for d in data_big:
	data_name = 'Data' + str(counter)	# to rename the sets again

	############################################################################################################################################
	# Fixation performance
	############################################################################################################################################

	fix = 'fix_performance'
	d[fix] = (d['fix_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['fix_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[fix] = d[fix] *1  # *1 makes true values 1, and false 0

	# Contrast 
	contrast = d.iloc[0]['pixstim_fix_opacity']
	contrasts.append(contrast)

	# Calculate performance + save performance in list
	pc_corr = round(((np.sum(d[fix]) / len(d))*100),2)
	performances.append(pc_corr)

	perf_info = str(data_name) + ' for ' + str(fix) + ' percent correct ' + str(pc_corr) + ' for contrast ' + str(contrast)	# save performance info in string
	performance_info.append(perf_info)


	# Print info
	print('> %s: %s = %s percent for contrast %s' %(data_name, fix, pc_corr, contrast))


	############################################################################################################################################
	# Background performance
	############################################################################################################################################

	bg = 'bg_performance'
	d[bg] = (d['bg_trial_stimulus_value'] < 0.0) & (d['answer'] == 1.0) | (d['bg_trial_stimulus_value'] > 0.0) & (d['answer'] == 0.0)
	d[bg] = d[bg] *1  # *1 makes true values 1, and false 0
	
	# Contrast + save contrast in list
	contrast = d.iloc[0]['pixstim_bg_opacity']
	contrasts.append(contrast)

	# Calculate performance + save performance in list
	pc_corr = round(((np.sum(d[bg]) / len(d))*100),2)
	performances.append(pc_corr)

	perf_info = str(data_name) + ' for ' + str(bg) + ' percent correct ' + str(pc_corr) + ' for contrast ' + str(contrast)	# save performance info in string
	performance_info.append(perf_info)

	# Print info
	print('> %s: %s = %s percent for contrast %s' %(data_name, bg, pc_corr, contrast))


	vars()[data_name] = d
	counter += 1

print('###################################################################')


########################################################################################################################################
# Make figures for datasets
########################################################################################################################################

figures = 1
stimuli = 'fix_trial_stimulus_value', 'bg_trial_stimulus_value'

subject_all_xdata = []
subject_all_ydata = []

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


		x_continuous = np.linspace(x_values.min(), x_values.max(), 100)
		y_continuous = sigmoid(x_continuous, *popt)

		# NEW
		subject_all_xdata.append(x_continuous)
		subject_all_ydata.append(y_continuous)



########################################################################################################################################
# Classification Matrix
########################################################################################################################################

#exp = np.random.random((4,4))

exp = np.array([[80.00, 80.00, 80.00, 80.00],
       [80.00, 70.00, 70.00, 50.00],
       [80.00 , 60.00 , 60.00, 50.00],
       [80.00, 50.00, 50.00, 50.00]])

plt.imshow(exp);

# X axis labels condition 1
xtick_val = [0, 1, 2, 3]		# location of ticks
xtick_lab = ['0.2','0.4','0.6', '0.8']		# label names of every tick
plt.xticks(xtick_val, xtick_lab)
plt.gca().set_xlabel('Contrast C1 Fixation')


# Y axis labels condition 2
ytick_val = [3, 2, 1, 0]		# location of ticks
ytick_lab = ['0.2','0.4','0.6', '0.8']		# label names of every tick
plt.yticks(ytick_val, ytick_lab)
plt.gca().set_ylabel('Contrast C2 Background')

plt.colorbar()
plt.show()













