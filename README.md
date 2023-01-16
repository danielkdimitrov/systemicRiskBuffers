# Macroprudential Regulation: A Risk Management Approach

## Optimal Systemic Risk Buffers


Here is a short decrption of the files in this repository. The file `main_runBuffersModels.py` is set up to run the models from the paper `Macroprudential Regulation: A Risk Management Approach` (Daniel Dimitrov; Sweder van Wijnbergen).

+ Follow the instructions within this file to run the model. 
+ You need daily or weekly data on CDS spreads for a universe of banks, as well as balance sheet information on the size of the liabilities of each bank. The data needs to be arranged in folder `data` following the same convention as the one provided. 

Here are the files and classes to which `main_runBuffersModels.py` refers to: 

1. `setParams.py` contains the main parameters of the model. Set here start and end date for the evaluation period. 

2. `DataLoad.py` contains the procedures loading and processing data. 

3. `GetImpliedParams.py` contains a customized preset procedure that refers to `DataLoad.py` for loading the data in a format that can be processed later on easily   

4. `myplotstyle.py` contains preset plot functions that are referred to in the paper. 

5. `optimalSystemicCapital` contains all the computational procedures in python class `PDmodel`. The class is organized in such a way that when you call it, you have to specify which model you want to run, e.g. the EEI model with `EEI opt w/data` or the Expected Shortfall minimization model with `min ES
