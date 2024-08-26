# WildfireABM

This is an agent based simulation of wildfire spread using python and the MESA library. A lot of the variables will seem obtuse, but they should be explained in the code, and they shouldn't really be changed, a large amount of them being constants. 

This is almost all based on the paper: `A cellular automata model for forest fire spread prediction: The case of the wildfire that swept through Spetses Island in 1990`, which is also `j.amc.2008.06.046`

To run: 
Install `requirements.txt` and then simply run `main.py` and a MESA server should open in your browser. When modifying variables, either the variable name should be self descriptive (e.g. pH, moisture, &c.) or the name will be highly obtuse. Ignore variable names that are obtuse, those are fine tuned constants.

## Example
![[Example video](https://cloud-k2fm6jy3o-hack-club-bot.vercel.app/0wildfireabm.png)](https://www.youtube.com/watch?v=m8k7ecnDySE)
