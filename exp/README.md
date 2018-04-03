# Run The PRF Experiment:
In the terminal, go to this folder. Then type ```python main.py <INITIALS>```, replacing `INITIALS` with your own initials. Later on, it will become ```python main.py <INITIALS> <INDEX>```, where `INDEX` stands for the type of condition to be run during this 'run'.

* Current version optimized (screen) for experimental lab setup (7T testroom)

## To do for behavioral pilots:


1. Staircase implementation for blobs and pixel conditions 
	- Are the key-responses of participant correctly updating the staircase values? 
		- Goes for both the 1u/1d and 3u/1d staircase!
	- Test blob conditions (fix and surround), and pixel conditions (fix and surround)
		- Make sure the correct intensity (bg or fix task) is updated and linked to responses

2. Add constant stimuli condition (could be run instead of staircase)
	- Create constant stimuli condition for determining optimal ratio red/green
		- Example idea: create array of values between 0.2 - 0.8, random select value (?) 

3. Output check of datafile:
	- Responses: are all the params saved correctly? Can we determine percentage correct/responsetimes and make psycometric curves to determine point of subjective equality? 
	- Staircase output check: can we reconstruct the entire staircase from the outputs in the pickle and edf files?

4. Is the timing of the pRF stimulus correct? No timing errors?

5. The spatial frequency content of the bg and fix stimuli need to be titrated to ensure global/local attentional focus 


## To do for scanning pilots:

1. Organize a pilot session to investigate voxel size and MB parameters
2. Decide on whether we want to use full 8 directions, or perhaps only 4? This would speed up the experiment tremendously, and allow us to have almost 2 times the amount of separate runs.

