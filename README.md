# FcsAnalysisPipeline

Authors: Alex Koch, James Bagnall, Hannah Perkins

A tool for analysing fluorescence correlation and cross-correlation spectroscopy files (.fcs) from the Zeiss ConfoCor3 FCS module. 

This tool comes as is and has not been extensively tested across different computers. It has run fine on machines with both Windows 7 & 10  as well as Linux (Ubuntu).

The working name for this project is Fish and Cushion.

Python package requirements to be installed via Pip (or by other means): 
- pandas
- numpy
- tqdm
- statsmodels
- scipy
- plotly
- colorlover
- tkinter
- json
- math
- os
- itertools
- datetime
- re
- warnings

In some cases plotly orca is required to output the plots in the browser with installation instructions here: https://github.com/plotly/orca

The fcsfiles package (Christoph Gohlke; v2019.1.1, https://pypi.org/project/fcsfiles/) is included as a minor change was made to ensure compatibility with our .fcs files. 

Once correctly installed the pipeline is inititiated by running fish_and_cushion.py using Python3 (v3.9 recommended). A window will pop up asking for the directory within which the .fcs files are contained. A list of all .fcs files found in the directory should be listed in the console window. You will then be promted to enter a name to append to the end of the outputted analysis and images folder that will be created in the directory containing the FCS files. Press enter to confirm this name. The program will then proceed to analyse the files as outlined by the settings file (setting.json). Once finished, provided that the option is "True" in the settings, a set of interactive sumary plots will pop up in your default browser alongside the analysis and images in the relevant folders. Press enter again in the console window to close the console was analysis is complete. 

Multiple models for each channel may be fit by specifying each model under the appropriate heading in the setting.json file followed by a comma, except for a lone or last entry. 

This tool is based on work by Sadaie et al 2014 (https://doi.org/10.1128/MCB.00087-14) for analysing interactions and Safarian and Elson 2003 (https://doi.org/10.1016/S0006-3495(03)75011-5) for estimating the standard deviation on correlation curves .
