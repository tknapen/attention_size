
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

		############################################################################
		# Find the x value for y value 0.5
		############################################################################

		y_stuff = np.round(y_continuous, 2)
		#cord = np.where(np.logical_and(y_stuff>=0.49, y_stuff<=0.51))
		cord = np.where(y_stuff == 0.5)

		# Catch error if finding x breaks
		try: # if y actually crosses 0, give x coordinate
			x_value = x_continuous[cord]			# array of x values
			x_value = round(x_value[0], 4,)			# changed to optimize saving in image, takes first value

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
		fig = plt.figure()
		plt.plot(x_values, y_values, 'o') #, label='data')
		plt.plot(x_continuous, y_continuous) # label='fit')
		plt.axhline(y=0.5, linestyle='dashed')
		plt.gca().set_ylim([-0.05,1.05])
		plt.gca().set_xlabel('Stim color (all red, 0, all green) | Crosses at x = %s' %(x_value))
		plt.gca().set_ylabel('% awnsered red')
		plt.gca().legend(loc='best')
		plt.gca().set_title('Performance for %s \n From %s | Contrast intensity: %s' %(s, subj, contrast))
		#plt.show()


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



########################################################################################################################################
# Make subimages with saved data
########################################################################################################################################

subject_all_xdata
subject_all_ydata
all_data = subject_all_xdata, subject_all_ydata

run1 = subject_all_xdata[0], subject_all_ydata[0]
run2 = subject_all_xdata[1], subject_all_ydata[1]
run3 = subject_all_xdata[2], subject_all_ydata[2]
run4 = subject_all_xdata[3], subject_all_ydata[3]
run5 = subject_all_xdata[4], subject_all_ydata[4]
run6 = subject_all_xdata[5], subject_all_ydata[5]
run7 = subject_all_xdata[6], subject_all_ydata[6]
run8 = subject_all_xdata[7], subject_all_ydata[7]
run9 = subject_all_xdata[8], subject_all_ydata[8]
run10 = subject_all_xdata[9], subject_all_ydata[9]
run11 = subject_all_xdata[10], subject_all_ydata[10]
run12 = subject_all_xdata[11], subject_all_ydata[11]
run13 = subject_all_xdata[12], subject_all_ydata[12]
run14 = subject_all_xdata[13], subject_all_ydata[13]
run15 = subject_all_xdata[14], subject_all_ydata[14]
run16 = subject_all_xdata[15], subject_all_ydata[15]
run17 = subject_all_xdata[16], subject_all_ydata[16]
run18 = subject_all_xdata[17], subject_all_ydata[17]
run19 = subject_all_xdata[18], subject_all_ydata[18]
run20 = subject_all_xdata[19], subject_all_ydata[19]

#plt.plot(run1[0], run1[1], label='Run1')
plt.plot(run2[0], run2[1], label='Run2')
#plt.plot(run3[0], run3[1], label='Run3')
plt.plot(run4[0], run4[1], label='Run4')
#plt.plot(run5[0], run5[1], label='Run5')
plt.plot(run6[0], run6[1], label='Run6')
#plt.plot(run7[0], run7[1], label='Run7')
plt.plot(run8[0], run8[1], label='Run8')
#plt.plot(run9[0], run9[1], label='Run9')
plt.plot(run10[0], run10[1], label='Run10')


#plt.plot(run11[0], run11[1], label='Run11')
plt.plot(run12[0], run12[1], label='Run12')
#plt.plot(run5[0], run5[1], label='Run5')
plt.plot(run6[0], run6[1], label='Run6')
#plt.plot(run7[0], run7[1], label='Run7')
plt.plot(run8[0], run8[1], label='Run8')
#plt.plot(run9[0], run9[1], label='Run9')
plt.plot(run10[0], run10[1], label='Run10')
#plt.plot(run9[0], run9[1], label='Run9')

plt.gca().set_ylabel('% awnsered red')
plt.gca().set_xlabel('Stim color (all red, 0, all green')
plt.gca().set_title('Performance of all trials for %s' %(subj))
plt.gca().legend(loc='best')
plt.show()



Figure plot all data
counter = 1
for allruns in range(0, len(subject_all_xdata)):
	xdata = subject_all_xdata[allruns]
	ydata = subject_all_ydata[allruns]

	plt.plot(xdata, ydata, label='Data ' + str(counter))
	plt.gca().set_ylabel('% awnsered red')
	plt.gca().set_xlabel('Stim color (all red, 0, all green')
	plt.gca().set_title('Performance of all trials for %s' %(subj))
	plt.gca().legend(loc='best')
	
	counter += 1

plt.show()














