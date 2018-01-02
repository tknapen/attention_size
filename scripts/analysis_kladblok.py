
####################################################################################################################################
# Python Analysis - Jutta de Jong
# Updated 9 December 2017
####################################################################################################################################

# Added:
# - Psychometric curves added for all 3 conditions + plotted figures

# - Prints mean color difference within correct/incorrect for relevant condition
# - First trial variables are all deleted (response times are weird etc.)


####################################################################################################################################

import sysconfig
sysconfig.ps1 = 'SOMETHING'                                                         # Fixes an interactive backend problem on my mac

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from pylab import get_current_fig_manager                                           # Otherwise figures open behind pycharm

import numpy as np
import copy
import seaborn as sn
sn.set(style="ticks")


####################################################################################################################################
# Open the saved data file
####################################################################################################################################

filename = 'subj03_run1_con1.tsv'

subject = (filename[4:6])
run = (filename[10])
condition = int(filename[15])
data_file = np.loadtxt(filename)
data_file = np.delete(data_file, 0, 0)  # Delete first trial

trial_numbers = data_file[:,0]
trial_duration = data_file[:,1]
colortask_response = data_file[:,2]
colortask_responsetimes = data_file[:,3]

condition1_amount_of_red = data_file[:,4]
condition1_amount_of_green = data_file[:,5]

condition2_amount_of_red = data_file[:,6]
condition2_amount_of_green = data_file[:,7]

condition3_amount_of_red = data_file[:,8]
condition3_amount_of_green = data_file[:,9]

amount_of_trials = float(len(trial_numbers))
print 'Subject %s' %subject,'| Run nr: %s' %run, '| Condition: %s' %condition, '| Total trials: %s' % int(amount_of_trials), '|'
print ('')


####################################################################################################################################
# Plot raw stuff
####################################################################################################################################

# plt.plot(trial_duration)
# plt.plot(colortask_responsetimes)


####################################################################################################################################
# Percentage correct for relevant conditions
####################################################################################################################################

def percentagecorrect(a, b):
    a = float(a)
    b = float(b)
    correctperc = a / b * 100
    correctperc = round(correctperc,2)
    return correctperc


####################################################################################################################################
# Condition 1
####################################################################################################################################

outcome_more_red_condition1 = []            # red (0,0,1..)
outcome_more_green_condition1 = []          # green (1,1,0..)
outcome_red_blobs_condition1 = []
outcome_green_blobs_condition1 = []

for x in range (0,len(trial_numbers)):

    redz = condition1_amount_of_red[x]
    outcome_red_blobs_condition1.append(redz)

    greenz = condition1_amount_of_green[x]
    outcome_green_blobs_condition1.append(greenz)

    if condition1_amount_of_red[x] > condition1_amount_of_green[x]:     # Most Red
        outcome_more_red_condition1.append(30)
        outcome_more_green_condition1.append(0)

    elif condition1_amount_of_green[x] > condition1_amount_of_red[x]:   # Most Green
        outcome_more_red_condition1.append(0)
        outcome_more_green_condition1.append(90)

    else:
        outcome_more_red_condition1.append(404)  # Equal
        outcome_more_green_condition1.append(404)


# Presence + response from participant
performance_condition1 = []

for y in range (0, len(colortask_response)):

    if colortask_response[y] == 30.0:                       # Response -> More Red present
        if outcome_more_red_condition1[y] == 30:            # More red present
            performance_condition1.append(1)
        elif outcome_more_green_condition1[y] == 90:        # More green present
            performance_condition1.append(0)
        else:
            performance_condition1.append(404)  # Equal

    elif colortask_response[y] == 90.0:                      # Response -> More Green present
        if outcome_more_red_condition1[y] == 30:
            performance_condition1.append(0)
        elif outcome_more_green_condition1[y] == 90:
            performance_condition1.append(1)
        else:
            performance_condition1.append(404)   # Equal

    elif colortask_response[y] == 0.0:
        performance_condition1.append(404)      # No response

    elif colortask_response[y] == 404:
        performance_condition1.append(404)

    else:
        print('broke')


# Filter out trial numbers that had equal colors or no response
clean_performance_condition1 = copy.deepcopy(performance_condition1)

no_res_word = 404
positions_c1 = []

if no_res_word in clean_performance_condition1:
    for i, j in enumerate(clean_performance_condition1):

        if j == no_res_word:
            positions_c1.append(i)


# Delete these trials from other variables as well
clean_outcome_more_red_condition1 = copy.deepcopy(outcome_more_red_condition1)
clean_outcome_more_green_condition1 = copy.deepcopy(outcome_more_green_condition1)

# Change numpy array's to lists
clean_condition1_amount_of_red = copy.deepcopy(condition1_amount_of_red.tolist())
clean_condition1_amount_of_green = copy.deepcopy(condition1_amount_of_green.tolist())


for x in range(0, len(positions_c1)):
    trial = positions_c1[x]
    del clean_performance_condition1[trial-x]
    del clean_outcome_more_red_condition1[trial-x]
    del clean_outcome_more_green_condition1[trial-x]
    del clean_condition1_amount_of_red[trial-x]
    del clean_condition1_amount_of_green[trial-x]


percentage_correct_condition1 = percentagecorrect(sum(clean_performance_condition1), len(clean_performance_condition1))
print('Percentage correct for condition 1: %s' % percentage_correct_condition1)


#####################################################################################################################################
# Condition 1 - Psychometrics
#####################################################################################################################################

# Should be percentage from total for one color only, NOT DIFFERENCE!
#psych_clean_difference = abs(np.subtract(clean_condition1_amount_of_green, clean_condition1_amount_of_red))

import pylab
from scipy.optimize import curve_fit

# Performance array
psych_clean_performance_c1 = np.asarray(clean_performance_condition1)

# Percentage green from total amount of blobs
psych_clean_xgreen_c1 = []
all_blobs = clean_condition1_amount_of_green[0] + clean_condition1_amount_of_red[0]

for gr in range (0,len(clean_condition1_amount_of_green)):
    green_blobs = clean_condition1_amount_of_green[gr]
    green_blobs = green_blobs/all_blobs*100
    psych_clean_xgreen_c1.append(green_blobs)

# Combine the two variables into list
psych = []
for x in range (0, len(psych_clean_performance_c1)):
    meh = psych_clean_performance_c1[x], psych_clean_xgreen_c1[x]
    psych.append(meh)

# Sort the list based on amount of green blobs
sort_psych1 = sorted(psych, key=lambda psych: psych[1])

# Change list to np array
sort_psych = np.array(sort_psych1)
sorted_psych_clean_xgreen_c1 = sort_psych[:,0]

sort_psych_clean_xgreen_c1 = sort_psych[:,1]
cum_sort_psych_clean_xgreen_c1 = np.hstack((0,sort_psych_clean_xgreen_c1))

# Cummulative scores from performance
cuml_sort_cl_pf_c1 = np.cumsum(sorted_psych_clean_xgreen_c1)
cuml_sort_cl_pf_c1/=cuml_sort_cl_pf_c1.max()
cuml_sort_cl_pf_c1 = np.hstack((0,cuml_sort_cl_pf_c1))

# Determine size of x axis based on x axis numbers
xl = len(cum_sort_psych_clean_xgreen_c1) - 1
x1 = cum_sort_psych_clean_xgreen_c1[0]
x2 = cum_sort_psych_clean_xgreen_c1[xl]

# Fit psychometric curve
xdata = np.array(cum_sort_psych_clean_xgreen_c1)
ydata = np.array(cuml_sort_cl_pf_c1)

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print popt

x = np.linspace(x1, x2, 10)
y = sigmoid(x, *popt)


plt.figure(1)
plt.plot(xdata, ydata, 'bo', label='C1')
plt.plot(x,y, 'b')
#pylab.plot(x,y, 'b',label='Fit C1')
plt.ylim(-0.05, 1.05)
plt.xlabel('Percentage green within total')
plt.ylabel('Cummulative correct responses')
plt.legend(loc='best')
plt.plot()



#####################################################################################################################################
# Condition 2
#####################################################################################################################################

outcome_more_red_condition2 = []
outcome_more_green_condition2 = []
outcome_red_blobs_condition2 = []
outcome_green_blobs_condition2 = []


for x in range (0,len(trial_numbers)):

    redz = condition2_amount_of_red[x]
    outcome_red_blobs_condition2.append(redz)

    greenz = condition2_amount_of_green[x]
    outcome_green_blobs_condition2.append(greenz)

    if condition2_amount_of_red[x] > condition2_amount_of_green[x]:     # Most Red
        outcome_more_red_condition2.append(30)
        outcome_more_green_condition2.append(0)

    elif condition2_amount_of_green[x] > condition2_amount_of_red[x]:   # Most Green
        outcome_more_red_condition2.append(0)
        outcome_more_green_condition2.append(90)

    else:
        outcome_more_red_condition2.append(404)  # Equal
        outcome_more_green_condition2.append(404)


# Presence + response from participant
performance_condition2 = []

for y in range (0, len(colortask_response)):

    if colortask_response[y] == 30.0:                       # Response -> More Red present
        if outcome_more_red_condition2[y] == 30:            # More red present
            performance_condition2.append(1)
        elif outcome_more_green_condition2[y] == 90:        # More green present
            performance_condition2.append(0)
        else:
            performance_condition2.append(404)  # Equal

    elif colortask_response[y] == 90.0:                      # Response -> More Green present
        if outcome_more_red_condition2[y] == 30:
            performance_condition2.append(0)
        elif outcome_more_green_condition2[y] == 90:
            performance_condition2.append(1)
        else:
            performance_condition2.append(404) # Equal

    elif colortask_response[y] == 0.0:
        performance_condition2.append(404) # No response

    elif colortask_response[y] == 404:
        performance_condition1.append(404)

    else:
        print('broke')


# Filter out trial numbers that had equal colors or no response
clean_performance_condition2 = copy.deepcopy(performance_condition2)

no_res_word = 404
positions_c2 = []

if no_res_word in clean_performance_condition2:
    for i, j in enumerate(clean_performance_condition2):

        if j == no_res_word:
            positions_c2.append(i)


# Delete these trials from other variables as well
clean_outcome_more_red_condition2 = copy.deepcopy(outcome_more_red_condition2)
clean_outcome_more_green_condition2 = copy.deepcopy(outcome_more_green_condition2)

# Change numpy array's to lists
clean_condition2_amount_of_red = copy.deepcopy(condition2_amount_of_red.tolist())
clean_condition2_amount_of_green = copy.deepcopy(condition2_amount_of_green.tolist())


for x in range(0, len(positions_c2)):
    trial = positions_c2[x]
    del clean_performance_condition2[trial-x]
    del clean_outcome_more_red_condition2[trial-x]
    del clean_outcome_more_green_condition2[trial-x]
    del clean_condition2_amount_of_red[trial-x]
    del clean_condition2_amount_of_green[trial-x]


percentage_correct_condition2 = percentagecorrect(sum(clean_performance_condition2), len(clean_performance_condition2))
print('Percentage correct for condition 2: %s' % percentage_correct_condition2)



#####################################################################################################################################
# Condition 2 - Psychometrics
#####################################################################################################################################

import pylab
from scipy.optimize import curve_fit

# Performance array
psych_clean_performance_c2 = np.asarray(clean_performance_condition2)

# Percentage green from total amount of blobs
psych_clean_xgreen_c2 = []
all_blobs = clean_condition2_amount_of_green[0] + clean_condition2_amount_of_red[0]

for gr in range (0,len(clean_condition2_amount_of_green)):
    green_blobs = clean_condition2_amount_of_green[gr]
    green_blobs = green_blobs/all_blobs*100
    psych_clean_xgreen_c2.append(green_blobs)

# Combine the two variables into list
psych_c2 = []
for x in range (0, len(psych_clean_performance_c2)):
    meh = psych_clean_performance_c2[x], psych_clean_xgreen_c2[x]
    psych_c2.append(meh)

# Sort the list based on amount of green blobs
psych_c2 = sorted(psych_c2, key=lambda psych_c2: psych_c2[1])

# Change list to np array
sort_psych_c2 = np.array(psych_c2)
sorted_psych_clean_xgreen_c2 = sort_psych_c2[:,0]

sort_psych_clean_xgreen_c2 = sort_psych_c2[:,1]
cum_sort_psych_clean_xgreen_c2 = np.hstack((0,sort_psych_clean_xgreen_c2))

# Cummulative scores from performance
cuml_sort_cl_pf_c2 = np.cumsum(sorted_psych_clean_xgreen_c2)
cuml_sort_cl_pf_c2/=cuml_sort_cl_pf_c2.max()                            # Normalise to a percentage
cuml_sort_cl_pf_c2 = np.hstack((0,cuml_sort_cl_pf_c2))

# Determine size of x axis based on x axis numbers
xl = len(cum_sort_psych_clean_xgreen_c2) - 1
x1 = cum_sort_psych_clean_xgreen_c2[0]
x2 = cum_sort_psych_clean_xgreen_c2[xl]

# Fit psychometric curve
xdata = np.array(cum_sort_psych_clean_xgreen_c2)
ydata = np.array(cuml_sort_cl_pf_c2)

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print popt

x = np.linspace(x1, x2, 10)
y = sigmoid(x, *popt)

plt.figure(2)
plt.plot(xdata, ydata, 'ro', label='C2')
plt.plot(x,y, 'r')
#pylab.plot(x,y, 'r',label='Fit C2')
plt.ylim(-0.05, 1.05)
plt.xlabel('Percentage green within total')
plt.ylabel('Cummulative correct responses')
plt.legend(loc='best')
plt.draw()



#####################################################################################################################################
# Condition 3
#####################################################################################################################################

outcome_more_red_condition3 = []
outcome_more_green_condition3 = []
outcome_red_blobs_condition3 = []
outcome_green_blobs_condition3 = []


for x in range (0,len(trial_numbers)):

    redz = condition3_amount_of_red[x]
    outcome_red_blobs_condition3.append(redz)

    greenz = condition3_amount_of_green[x]
    outcome_green_blobs_condition3.append(greenz)

    if condition3_amount_of_red[x] > condition3_amount_of_green[x]:     # Most Red
        outcome_more_red_condition3.append(30)
        outcome_more_green_condition3.append(0)

    elif condition3_amount_of_green[x] > condition3_amount_of_red[x]:   # Most Green
        outcome_more_red_condition3.append(0)
        outcome_more_green_condition3.append(90)

    else:
        outcome_more_red_condition3.append(404)  # Equal
        outcome_more_green_condition3.append(404)


# Presence + response from participant
performance_condition3 = []

for y in range (0, len(colortask_response)):

    if colortask_response[y] == 30.0:                       # Response -> More Red present
        if outcome_more_red_condition3[y] == 30:            # More red present
            performance_condition3.append(1)
        elif outcome_more_green_condition3[y] == 90:        # More green present
            performance_condition3.append(0)
        else:
            performance_condition3.append(404)  # Equal

    elif colortask_response[y] == 90.0:                      # Response -> More Green present
        if outcome_more_red_condition3[y] == 30:
            performance_condition3.append(0)
        elif outcome_more_green_condition3[y] == 90:
            performance_condition3.append(1)
        else:
            performance_condition3.append(404) # Equal

    elif colortask_response[y] == 0.0:
        performance_condition3.append(404) # No response

    elif colortask_response[y] == 404:
        performance_condition1.append(404)

    else:
        print('broke')


# Filter out trial numbers that had equal colors or no response
clean_performance_condition3 = copy.deepcopy(performance_condition3)

no_res_word = 404
positions_c3 = []

if no_res_word in clean_performance_condition3:
    for i, j in enumerate(clean_performance_condition3):

        if j == no_res_word:
            positions_c3.append(i)


# Delete these trials from other variables as well
clean_outcome_more_red_condition3 = copy.deepcopy(outcome_more_red_condition3)
clean_outcome_more_green_condition3 = copy.deepcopy(outcome_more_green_condition3)

# Change numpy array's to lists
clean_condition3_amount_of_red = copy.deepcopy(condition3_amount_of_red.tolist())
clean_condition3_amount_of_green = copy.deepcopy(condition3_amount_of_green.tolist())


for x in range(0, len(positions_c3)):
    trial = positions_c3[x]
    del clean_performance_condition3[trial-x]
    del clean_outcome_more_red_condition3[trial-x]
    del clean_outcome_more_green_condition3[trial-x]
    del clean_condition3_amount_of_red[trial-x]
    del clean_condition3_amount_of_green[trial-x]


percentage_correct_condition3 = percentagecorrect(sum(clean_performance_condition3), len(clean_performance_condition3))
print('Percentage correct for condition 3: %s' % percentage_correct_condition3)



#####################################################################################################################################
# Condition 3 - Psychometrics
#####################################################################################################################################

import pylab
from scipy.optimize import curve_fit

# Performance array
psych_clean_performance_c3 = np.asarray(clean_performance_condition3)

# Percentage green from total amount of blobs
psych_clean_xgreen_c3 = []
all_blobs = clean_condition3_amount_of_green[0] + clean_condition3_amount_of_red[0]

for gr in range (0,len(clean_condition3_amount_of_green)):
    green_blobs = clean_condition3_amount_of_green[gr]
    green_blobs = green_blobs/all_blobs*100
    psych_clean_xgreen_c3.append(green_blobs)

# Combine the two variables into list
psych_c3 = []
for x in range (0, len(psych_clean_performance_c3)):
    meh = psych_clean_performance_c3[x], psych_clean_xgreen_c3[x]
    psych_c3.append(meh)

# Sort the list based on amount of green blobs
psych_c3 = sorted(psych_c3, key=lambda psych_c3: psych_c3[1])

# Change list to np array
sort_psych_c3 = np.array(psych_c3)
sorted_psych_clean_xgreen_c3 = sort_psych_c3[:,0]

sort_psych_clean_xgreen_c3 = sort_psych_c3[:,1]
cum_sort_psych_clean_xgreen_c3 = np.hstack((0,sort_psych_clean_xgreen_c3))

# Cummulative scores from performance
cuml_sort_cl_pf_c3 = np.cumsum(sorted_psych_clean_xgreen_c3)
cuml_sort_cl_pf_c3/=cuml_sort_cl_pf_c3.max()                            # Normalise to a percentage
cuml_sort_cl_pf_c3 = np.hstack((0,cuml_sort_cl_pf_c3))

# Determine size of x axis based on x axis numbers
xl = len(cum_sort_psych_clean_xgreen_c3) - 1
x1 = cum_sort_psych_clean_xgreen_c3[0]
x2 = cum_sort_psych_clean_xgreen_c3[xl]

# Fit psychometric curve
xdata = np.array(cum_sort_psych_clean_xgreen_c3)
ydata = np.array(cuml_sort_cl_pf_c3)

def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print popt

x = np.linspace(x1, x2, 10)
y = sigmoid(x, *popt)

plt.figure(3)
plt.plot(xdata, ydata, 'go', label='C3')
plt.plot(x,y, 'g')
#pylab.plot(x,y, 'r',label='Fit C3')
plt.ylim(-0.05, 1.05)
plt.xlabel('Percentage green within total')
plt.ylabel('Cummulative correct responses')
plt.legend(loc='best')
plt.plot()




#####################################################################################################################################
# Color Difference related to task performance, relevant condition
#####################################################################################################################################

most_red = []
most_green = []
difference_between_colors = []
correct_trial_color_difference = []
incorrect_trial_color_difference = []

for x in range (0,1):

    if condition == 1:

        #First cut out the previous removed trials
        clean_amount_of_red_condition1 = copy.deepcopy(outcome_red_blobs_condition1)
        clean_amount_of_green_condition1 = copy.deepcopy(outcome_green_blobs_condition1)

        for x in range(0, len(positions_c1)):
            trial = positions_c1[x]
            del clean_amount_of_red_condition1[trial-x]
            del clean_amount_of_green_condition1[trial-x]

        for scores in range(0, len(clean_performance_condition1)):
            if clean_amount_of_red_condition1[scores] < clean_amount_of_green_condition1[scores]:   # More red blobs
                diff = clean_amount_of_green_condition1[scores] - clean_amount_of_red_condition1[scores]
                difference_between_colors.append(diff)
                most_red.append(1)
                most_green.append(0)

            elif clean_amount_of_green_condition1[scores] < clean_amount_of_red_condition1[scores]:   # More green blobs
                diff = clean_amount_of_red_condition1[scores] - clean_amount_of_green_condition1[scores]
                difference_between_colors.append(diff)
                most_red.append(0)
                most_green.append(1)

        for scores in range(0, len(clean_performance_condition1)):
            if clean_performance_condition1[scores] == 1:  # Correct trial
                difference = difference_between_colors[scores]
                correct_trial_color_difference.append(difference)  # Append difference scores

            elif clean_performance_condition1[scores] == 0:  # Incorrect trial
                difference = difference_between_colors[scores]
                incorrect_trial_color_difference.append(difference)  # Append difference scores

            else:
                print('Whoops something broke con1')


    if condition == 2:
        # First cut out the previous removed trials
        clean_amount_of_red_condition2 = copy.deepcopy(outcome_red_blobs_condition2)
        clean_amount_of_green_condition2 = copy.deepcopy(outcome_green_blobs_condition2)

        for x in range(0, len(positions_c2)):
            trial = positions_c2[x]
            del clean_amount_of_red_condition2[trial - x]
            del clean_amount_of_green_condition2[trial - x]

        for scores in range(0, len(clean_performance_condition2)):
            if clean_amount_of_red_condition2[scores] < clean_amount_of_green_condition2[scores]:  # More red blobs
                        diff = clean_amount_of_green_condition2[scores] - clean_amount_of_red_condition2[scores]
                        difference_between_colors.append(diff)
                        most_red.append(1)
                        most_green.append(0)

            elif clean_amount_of_green_condition2[scores] < clean_amount_of_red_condition2[scores]:  # More green blobs
                        diff = clean_amount_of_red_condition2[scores] - clean_amount_of_green_condition2[scores]
                        difference_between_colors.append(diff)
                        most_red.append(0)
                        most_green.append(1)

        for scores in range(0, len(clean_performance_condition2)):

            if clean_performance_condition2[scores] == 1:  # Correct trial
                difference = difference_between_colors[scores]
                correct_trial_color_difference.append(difference)  # Append difference scores

            elif clean_performance_condition2[scores] == 0:  # Incorrect trial
                difference = difference_between_colors[scores]
                incorrect_trial_color_difference.append(difference)  # Append difference scores

            else:
                print('Whoops something broke con2')


    if condition == 3:
        # First cut out the previous removed trials
        clean_amount_of_red_condition3 = copy.deepcopy(outcome_red_blobs_condition3)
        clean_amount_of_green_condition3 = copy.deepcopy(outcome_green_blobs_condition3)

        for x in range(0, len(positions_c3)):
            trial = positions_c3[x]
            del clean_amount_of_red_condition3[trial - x]
            del clean_amount_of_green_condition3[trial - x]

        for scores in range(0, len(clean_performance_condition3)):
            if clean_amount_of_red_condition3[scores] < clean_amount_of_green_condition3[scores]:  # More red blobs
                diff = clean_amount_of_green_condition3[scores] - clean_amount_of_red_condition3[scores]
                difference_between_colors.append(diff)
                most_red.append(1)
                most_green.append(0)

            elif clean_amount_of_green_condition3[scores] < clean_amount_of_red_condition3[scores]:  # More green blobs
                diff = clean_amount_of_red_condition3[scores] - clean_amount_of_green_condition3[scores]
                difference_between_colors.append(diff)
                most_red.append(0)
                most_green.append(1)

        for scores in range(0, len(clean_performance_condition3)):
            if clean_performance_condition3[scores] == 1:  # Correct trial
                difference = difference_between_colors[scores]
                correct_trial_color_difference.append(difference)  # Append difference scores

            elif clean_performance_condition3[scores] == 0:  # Incorrect trial
                difference = difference_between_colors[scores]
                incorrect_trial_color_difference.append(difference)  # Append difference scores

            else:
                print('Whoops something broke c3')



print''
correct_diff_mean = round(float(np.mean(correct_trial_color_difference)),2)
correct_diff_stdev = round(float(np.std(correct_trial_color_difference)),2)
print 'For relevant condition: color difference correct trials: M = %s' % correct_diff_mean, 'SD = %s' % correct_diff_stdev

incorrect_diff_mean = round(float(np.mean(incorrect_trial_color_difference)),2)
incorrect_diff_stdev = round(float(np.std(incorrect_trial_color_difference)),2)
print 'For relevant condition: incorrect trials: M = %s' % incorrect_diff_mean, 'SD = %s' % incorrect_diff_stdev



#####################################################################################################################################
# Relating performance to responsetimes
#####################################################################################################################################

clean_responsetimes = []
correct_trial_responsetimes = []
incorrect_trial_responsetimes = []


# Cut first trial off (because weird delay when startup?)??


for x in range (0,1):

        if condition == 1:

            for x in range(0, len(colortask_responsetimes)):
                time = abs(colortask_responsetimes[x])
                clean_responsetimes.append(time)

            for x in range(0, len(positions_c1)):                                   # Remove  previously filtered trials
                trial = positions_c1[x]
                del clean_responsetimes[trial-x]

            for scores in range (0,len(clean_responsetimes)):
                if clean_performance_condition1[scores] == 1:                       # Correct trial
                    responsetime = clean_responsetimes[scores]
                    correct_trial_responsetimes.append(responsetime)

                elif clean_performance_condition1[scores] == 0:                     # Incorrect trial
                    responsetime = clean_responsetimes[scores]
                    incorrect_trial_responsetimes.append(responsetime)

                else:
                    print('Whoops something broke responsetimes c1')


        if condition == 2:

            for x in range(0, len(colortask_responsetimes)):
                time = abs(colortask_responsetimes[x])
                clean_responsetimes.append(time)

            for x in range(0, len(positions_c2)):  # Remove the previously filtered trials
                trial = positions_c2[x]
                del clean_responsetimes[trial - x]

            for scores in range(0, len(clean_responsetimes)):
                if clean_performance_condition2[scores] == 1:  # Correct trial
                    responsetime = clean_responsetimes[scores]
                    correct_trial_responsetimes.append(responsetime)

                elif clean_performance_condition2[scores] == 0:  # Incorrect trial
                    responsetime = clean_responsetimes[scores]
                    incorrect_trial_responsetimes.append(responsetime)

                else:
                    print('Whoops something broke responsetimes c2')



        if condition == 3:

            for x in range(0, len(colortask_responsetimes)):
                time = abs(colortask_responsetimes[x])
                clean_responsetimes.append(time)

            for x in range(0, len(positions_c3)):  # Remove the previously filtered trials
                trial = positions_c3[x]
                del clean_responsetimes[trial - x]

            for scores in range(0, len(clean_responsetimes)):
                if clean_performance_condition3[scores] == 1:  # Correct trial
                    responsetime = clean_responsetimes[scores]
                    correct_trial_responsetimes.append(responsetime)

                elif clean_performance_condition3[scores] == 0:  # Incorrect trial
                    responsetime = clean_responsetimes[scores]
                    incorrect_trial_responsetimes.append(responsetime)

                else:
                    print('Whoops something broke responsetimes c3')


print # white line
correct_rt_mean = int(round(float(np.mean(correct_trial_responsetimes)),2) * 1000)
correct_rt_stdev = int(round(float(np.std(correct_trial_responsetimes)),2) * 1000)
print'RT correct trials| M = %s' % correct_rt_mean, 'ms| SD = %s' % correct_rt_stdev, 'ms |'

incorrect_rt_mean = int(round(float(np.mean(incorrect_trial_responsetimes)),2) * 1000)
incorrect_rt_stdev = int(round(float(np.std(incorrect_trial_responsetimes)),2) * 1000)
print'RT incorrect trials| M = %s' % incorrect_rt_mean, 'ms| SD = %s' % incorrect_rt_stdev, 'ms |'



#####################################################################################################################################
# Plotting results
#####################################################################################################################################

plt.figure(4)
plt.figure(4).canvas.manager.window.attributes('-topmost', 1)

index = np.arange(1)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, percentage_correct_condition1, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Condition 1')

rects2 = plt.bar(index + bar_width+0.05, percentage_correct_condition2, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Condition 2')


rects3 = plt.bar(index + bar_width + bar_width +0.10, percentage_correct_condition3, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='Condition 3')

axes = plt.gca()
axes.set_ylim([0,100])
plt.xticks([])          # Turn off x axis labels
plt.xlabel('Conditions')
plt.ylabel('Percentage Correct')
plt.legend()
plt.tight_layout()
plt.show()


#####################################################################################################################################
# Correlate performance with color difference
#####################################################################################################################################
#
# clean_performance = []
#
# if condition == 1:
#     clean_performance = clean_performance_condition1
# elif condition == 2:
#     clean_performance = clean_performance_condition2
# elif condition ==3:
#     clean_performance = clean_performance_condition3
# else:
#     'performance x difference correlation plot crashed'
#
# x = clean_performance
# y =  difference_between_colors
# plt.plot(x, y, 'o')
# #plt.xticks([])
# plt.xlabel('Incorrect trial, correct trial')
# plt.ylabel('Difference between red and green')
#
#
# plt.axis([-0.25, 1.25, 0, 10])
# plt.show()
#

#####################################################################################################################################

# # generate some fake data
# N = 50
# x = np.random.randn(N, 1)
# y = x*2.2 + np.random.randn(N, 1)*0.4 - 1.8
# #plt.axhline(0, color='r', zorder=-1)
# #plt.axvline(0, color='r', zorder=-1)
# plt.scatter(x, y)
# #
# # fit least-squares with an intercept
# w = np.linalg.lstsq(np.hstack((x, np.ones((N,1)))), y)[0]
# xx = np.linspace(*plt.gca().get_xlim()).T
#
# # plot best-fit line
# plt.plot(xx, w[0]*xx + w[1], '-k')
