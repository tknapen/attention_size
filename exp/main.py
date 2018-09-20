import sys
from session import AttSizeSession
import appnope


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


    ts = AttSizeSession(subject_initials=initials, index_number=index_number, task=task, tracker_on=False)          # Normal Experiment
    #ts = AttSizeSessionFF(subject_initials=initials, index_number=index_number, task=task, tracker_on=False)          # Flicker Fusion

    ts.run()

if __name__ == '__main__':
    main()
