# Run The PRF Experiment:
In the terminal, go to this folder. Then type ```python main.py <INITIALS>```, replacing `INITIALS` with your own initials. Later on, it will become ```python main.py <INITIALS> <INDEX> <TASK>```, where `INDEX` stands for the staircase type (1u/1d or 3u/1d) and `TASK` for which condition (fixation or surround) to be run during this 'run'.

* Current version optimized (screen) for experimental lab setup (7T testroom)

## To do for behavioral pilots:

1. Two staircases for each (fixation and surround condition): 
    * One-Up-One-Down staircase to find point of subjective equality (equiluminance) for red and green.
    * Three-Up-One-Down staircase to titrate difficulty in 2-AFC task
    * *Store staircases in a dictionary to be pickled?*
    * *Print outcome staircase + total percentage correct after each run?*

2. Fixation conditions (for blob and pixel) with seperate params read-in from json file
	* Size of fixation condition can be/is coupled to fixation aperture size (to keep the size flexible)
	* Already present for pixel condition

3. If 1u/1d staircases are selected, don't draw the pPRF stimulus
4. The spatial frequency content of the bg and fix stimuli need to be titrated to ensure global/local attentional focus. 
5. Also, the opacity of the background stimulus needs to be set.
6. Test outputs; can we reconstruct the entire staircase from the outputs in the pickle and edf files?
    * Psychometric curve of performance per run

7. Double check if timing pRF stimulus is correct / doesn't accumelate error / too many frames over run
8. Optional: add 2 borders around fix and surround condition, for blob and pixel version

## To do for scanning pilots:

1. Organize a pilot session to investigate voxel size and MB parameters
2. Decide on whether we want to use full 8 directions, or perhaps only 4? This would speed up the experiment tremendously, and allow us to have almost 2 times the amount of separate runs.

