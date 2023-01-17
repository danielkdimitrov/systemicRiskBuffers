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

## How to run the model

1. Update the data in a folder `data` keeping the the sample formats attached in this repository. The following files need to be updated

  a. In folder `cds` update the time series of the CDS prices for all banks. Note that currently the CDS files contain only sample data, as the original data is Bloomberg proprietary. 

  b. In folder `debt` update the size of the liabilities of each bank. 

  c. In folder `other` update the `BankDefinitions.csv` file. In the column `Sample` indicate that the bank will be part of consequent analysis with a `Y`. Including other columns is optional. 
  
2. In file `setParams.py` set the parameters `firstDate` and `lastDate` which will determine the time window used for the analysis. Further options can be changed here, such as the number of factors (`n`), and number of simulations (`nSims`) 

3. Run file `main_runBuffersModels.py`. There are three sections in the file

  a. Import data: gets the input data in the appropriate format
  
  b. Run Expected Equal Impact Model
  
  c. Runnin ES model subject to average target

Once the appropriate model is run, the output data is arranged as parts of the output class `myPD`. The file `main_runBuffersModels` provides sample output and charts. 

For questions and comments, contact me at daniel.k.dimitrov@gmail.com. 
