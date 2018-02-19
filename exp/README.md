# Run The PRF Experiment:
In the terminal, go to this folder. Then type ```python main.py <INITIALS>```, replacing `INITIALS` with your own initials. Later on, it will become ```python main.py <INITIALS> <INDEX>```, where `INDEX` stands for the type of condition to be run during this 'run'.

## To do for behavioral pilots:

1. Separate staircases for fixation mark and background stimulus.
2. Two staircases for each (fix and bg): 
    * One-Up-One-Down staircase to find point of subjective equality (equiluminance) for red and green.
    * Three-Up-One-Down staircase to titrate difficulty in 2-AFC task
    * *Store staircases in a dictionary to be pickled?*
3. The spatial frequency content of the bg and fix stimuli need to be titrated to ensure global/local attentional focus. 
4. Also, the opacity of the background stimulus needs to be set.
5. Fixation apperture was added for safety zone, but minor change required for showing instruction text 
6. For 1/F stim, add fixation condition
7. Add 2(?) borders around conditions, for both exp versions?
8. Test outputs; can we reconstruct the entire staircase from the outputs in the pickle and edf files?


## To do for scanning pilots:

1. Organize a pilot session to investigate voxel size and MB parameters
2. Decide on whether we want to use full 8 directions, or perhaps only 4? This would speed up the experiment tremendously, and allow us to have almost 2 times the amount of separate runs.

