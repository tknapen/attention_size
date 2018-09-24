import sys
import os
import json
from session import AttSizeSession, AttSizeSessionFF
import appnope


##################################################################################
# Get and open json file
##################################################################################

config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')
with open(config_file) as config_file:
            config = json.load(config_file)


##################################################################################
# Run main
##################################################################################

def main():
    initials = sys.argv[1]
    # raw_input('Your initials: ')
    task = int(sys.argv[2])
    #scanner = raw_input('Are you in the scanner (y/n)?: ')
    #track_eyes = raw_input('Are you recording gaze (y/n)?: ')
    # if track_eyes == 'y':
        #tracker_on = True
    # elif track_eyes == 'n':
        #tracker_on = False
    # task = int(raw_input('fixation (0) or surround (1) task?: '))

    # initials = 'tk'
    index_number = int(sys.argv[3])        
    # task = 0                # 0 is fixation task, 1 is surround task
    appnope.nope()


    # Normal Experiment
    if config['flickerfuse'] == 0: 
        ts = AttSizeSession(subject_initials=initials, index_number=index_number, task=task, tracker_on=False)      
        print('##### Main - Runs normal Experiment #####')   
    
    # Flicker Fusion threshold
    else:
        ts = AttSizeSessionFF(subject_initials=initials, index_number=index_number, task=task, tracker_on=False)          
        print('##### Main - Flicker Fuse Test #####')   


    ts.run()

if __name__ == '__main__':
    main()
