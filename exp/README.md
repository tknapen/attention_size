# Run The PRF Experiment:
In the terminal, go to this folder. Then type ```python main.py <INITIALS>```, replacing `INITIALS` with your own initials. Later on, it will become ```python main.py <INITIALS> <INDEX>```, where `INDEX` stands for the type of condition to be run during this 'run'.

* Current version optimized (screen) for experimental lab setup (7T testroom)

## To do for behavioral pilots:

1. New working masks for fixation and surround (so stimuli can be used as seperate objects)
   - Draw order of the masks and stimuli are incorrect (include masks into stim objects, i.e. recalculate_stim tweak opacity param??)
   - Fixation condition is too large, should be reduced to smaller size (include seperate size params for basic_textures function??)

2. Blob stimulus: add fixation condition and think about layout (1/2 circles of many(!) blobs, with or without fixation cross?)
3. Staircase output check: can we reconstruct the entire staircase from the outputs in the pickle and edf files?
4. The spatial frequency content of the bg and fix stimuli need to be titrated to ensure global/local attentional focus. 
5. Opacity of the background stimulus needs to be set.
6. Add 2(?) borders around fixation and surround conditions, for both exp versions


## To do for scanning pilots:

1. Organize a pilot session to investigate voxel size and MB parameters
2. Decide on whether we want to use full 8 directions, or perhaps only 4? This would speed up the experiment tremendously, and allow us to have almost 2 times the amount of separate runs.

