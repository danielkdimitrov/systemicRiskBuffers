# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:46:26 2023

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from optimalSystemicCapital import PDmodel
from myplotstyle import * 
#from GetSystemicRiskSims import * # ??
from GetImpliedParams import *

from datetime import timedelta

from scipy.stats import norm
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest

import pickle

'''     
This code runs the examples 

myPD_sigma10 = PDmodel(.1)
myPD_sigma15 = PDmodel(.15)            
            
      
plt.plot(myPD_sigma10.k_grid, myPD_sigma10.PD*100,label=r'$\sigma=.1$')
plt.plot(myPD_sigma10.k_grid, myPD_sigma15.PD*100,label=r'$\sigma=.15$')
plt.legend()
plt.xlabel('k')        
plt.ylabel('PD (%)')        

'''


path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\imagesTemp\\"

#%%%
'unpickle'    
with open('saved_dictionaryEURO01.pkl', 'rb') as f:
    paramsDict = pickle.load(f)


#%% --------------- Import And Save Data ---------------------
'Get parameters'

mySRparams = GetImpliedParams(dutchSubsample = False, fixedRRs = True)

'filter out banks not in the sample, e.g. NIBC'
myDataset = mySRparams.DataSet.banks[mySRparams.DataSet.banks.Sample == 'Y']
myDataset = myDataset.join(mySRparams.dfSigmaEval.T, on='Short Code') 
myDataset.rename(columns={myDataset.columns[-1]: "Sigma"}, inplace=True)

myDataset['p2r  CET1']['DANK'] = myDataset['p2r  CET1'].median()

paramsDict = {}
paramsDict['Sigma'], paramsDict['wts'], paramsDict['LGD'], paramsDict['Rho'], paramsDict['Names'] = \
    mySRparams.dfSigmaEval.values.astype(np.float64), mySRparams.wtsEval.values.astype(np.float64), mySRparams.LGDEval.values.astype(np.float64), mySRparams.ldngs.astype(np.float64), mySRparams.dfSigmaEval.columns
paramsDict['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']].values.T[0]

'including pillar 2:'
mySRparams.DataSet.banks['p2r  CET1']['DANK'] = mySRparams.DataSet.banks['p2r  CET1'].median()
paramsDict['k_p2r'] = mySRparams.DataSet.banks['p2r  CET1'][mySRparams.universe].values.astype(np.float64)


'''
'pickle'
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(paramsDict, f)
'''

'Table with descriptive statistics'
table = pd.DataFrame(index=paramsDict['Names'], columns = ['Country','Code','Name','w','CDS', 'PD','rho1', 'rho2', 'rho3','sigma_hat', 'k', 'k_p2r']) #, 'k_macro'
j = 0
for index, row in table.iterrows():
    print(j)
    row['Country'] = mySRparams.DataSet.banks.loc[index,'Country']
    row['Code'] = index
    row['Name'] =mySRparams.DataSet.banks.loc[index,'Bank Name']
    row['w'] = paramsDict['wts'][j]*100
    row['CDS'] = mySRparams.DataSet.CDSprices.loc[mySRparams.evalDate,index].values[0]
    row['PD'] = mySRparams.dfPD.loc[mySRparams.evalDate,index].values[0]*100
    #row['LGD'] = paramsDict['LGD'][j]*100
    row['sigma_hat'] = paramsDict['Sigma'].T[j][0]*100
    row['rho1'] = paramsDict['Rho'][j][0]
    row['rho2'] = paramsDict['Rho'][j][1]
    row['rho3'] = paramsDict['Rho'][j][2]
    row['k'] = mySRparams.DataSet.capitalRatio.loc[mySRparams.evalDate,index].values[0]
    row['k_p2r'] = paramsDict['k_p2r'][j]*100
    #row['k_macro'] = myPD.dict['k_macro_str'][j]
    j = j +1 


table.sort_values(by ='Country').to_excel('rawData_80LGD_v02.xlsx')
print(table.round(2).sort_values(by ='Country').to_latex())    

