# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:32:05 2023

@author: danie
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from optimalSystemicCapital import PDmodel
from myplotstyle import * 
from GetImpliedParams import *

from datetime import timedelta
from scipy.stats import norm
import pickle

#%%'0. ---------------------- Import data ---------------------- '
mySRparams = GetImpliedParams(dutchSubsample = True, fixedRRs = True)

#%%
'1. ---------------------- Run Expected Equal Impact Model ----------------------'

paramsDict = {}
paramsDict['Sigma'], paramsDict['wts'], paramsDict['LGD'], paramsDict['Rho'], paramsDict['Names'] = \
    mySRparams.dfSigmaEval.values.astype(np.float64), mySRparams.wtsEval.values.astype(np.float64), mySRparams.LGDEval.values.astype(np.float64), mySRparams.ldngs.astype(np.float64), mySRparams.dfSigmaEval.index
paramsDict['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']].values.T[0]

'Input the target loss L_bar here ): '
#paramsDict['k_bar'] = 0.073684 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
paramsDict['Lbar'] = .0

'including pillar 2:'
mySRparams.DataSet.banks['p2r  CET1']['DANK'] = mySRparams.DataSet.banks['p2r  CET1'].median()
paramsDict['k_p2r'] = mySRparams.DataSet.banks['p2r  CET1'][mySRparams.universe].values.astype(np.float64)

'::::: Run model :::::'
myPD = PDmodel('EEI opt w/data', paramsDict)

'Model Output : '
print('EGD_ref=', myPD.EAD_ref)
print('SCD_ref',myPD.SCD_ref)
print('SCD',myPD.dict['SCD'])
print('IndirectCost',myPD.dict['IndirectCost'])
print('DirectCost',myPD.dict['DirectCost'])
print('PD',myPD.dict['PD'])
print('k_macro_str',myPD.dict['k_macro_str'])
print('--------- Univariate -----------')
print('k_macro_str_univ',myPD.dict['k_macro_str_univ'])
print('SCD_univ',myPD.dict['SCD_univ'])

'some plots'
plt.bar(paramsDict['Names'],myPD.dict['k_macro_str'])
plt.bar(paramsDict['Names'],paramsDict['wts'] )
plt.bar(paramsDict['Names'], myPD.dict['SCD'] )
plt.bar(paramsDict['Names'], myPD.dict['DirectCost'] )
plt.bar(paramsDict['Names'], myPD.dict['IndirectCost'] )


plt.bar(paramsDict['Names'], myPD.dict['SCD_univ'] )

X_axis = np.arange(len(paramsDict['Names']))
plt.bar(X_axis-.2, myPD.dict['k_macro_str_univ'],  0.4, label='univar' )
plt.bar(X_axis+.2, myPD.dict['k_macro_str'],  0.4, label='multivar' )
plt.legend()
plt.xticks(X_axis, paramsDict['Names'])

X_axis = np.arange(len(paramsDict['Names']))
plt.bar(X_axis-.2, myPD.dict['SCD_univ'],  0.4, label='univar' )
plt.bar(X_axis+.2, myPD.dict['SCD'],  0.4, label='multivar' )
plt.legend()
plt.xticks(X_axis, paramsDict['Names'])



'---------------------- Runnin ES model subject to average target ----------------------'

mySRparams = GetImpliedParams(dutchSubsample = False, fixedRRs = True)

paramsDict = {}
paramsDict['Sigma'], paramsDict['wts'], paramsDict['LGD'], paramsDict['Rho'], paramsDict['Names'] = \
    mySRparams.dfSigmaEval.values.astype(np.float64), mySRparams.wtsEval.values.astype(np.float64), mySRparams.LGDEval.values.astype(np.float64), mySRparams.ldngs.astype(np.float64), mySRparams.dfSigmaEval.index
paramsDict['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']].values.T[0]

'Now the minization runs against the average O-SII rate in the sample.'
paramsDict['k_bar'] = np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
paramsDict['Lbar'] = .0

'including pillar 2:'
mySRparams.DataSet.banks['p2r  CET1']['DANK'] = mySRparams.DataSet.banks['p2r  CET1'].median()
paramsDict['k_p2r'] = mySRparams.DataSet.banks['p2r  CET1'][mySRparams.universe].values.astype(np.float64)


'Run the minimization'
myPD = PDmodel('min ES', paramsDict, True)


'Collect output in a single table'
dfTable = pd.DataFrame(index =paramsDict['Names'], columns = ['Country','O-SII Rate', 'k_macro_str'])
dfTable['k_macro_str'] = myPD.dict['k_macro_str']*100 
dfTable['O-SII Rate'] =  paramsDict['O-SII rates']*100
dfTable['k_i'] = mySRparams.DataSet.capitalRatio.iloc[0]
dfTable['k_i_str'] = myPD.dict['k_str'] 
dfTable['Country'] = mySRparams.DataSet.banks.loc[dfTable.index,'Country']

print(dfTable)

'Export in excel'
dfTable.sort_values(by='Country').to_excel('output_minESeuro3atOSII.xlsx') #output_minESeuro3atCurrentKbar_v02

'Do a bar plot of current vs. optimal k'
dfTable[['k_i','k_i_str']].plot.bar(figsize = (9,4))

dfTable.sort_values(by='k_macro_str',ascending=False).plot.bar(figsize=(12,6), cmap = 'tab20b')
plt.legend(['O-SII Rate',r'$k_{macro,i}^*$'])
plt.xlabel('')



'3. ---------------------- Determine Socially Optimal k_bar ----------------------'

'Define a function evaluating Social Disutility at different k_bar. Parameters of the model are hardcoded here, and can be changed within this function '
def getECostKmicro(k_micro):

    paramsDict['Lbar'] = .4
    K_bar = np.linspace(0,.2,20) #  0.05,.1
    
    dfES = pd.DataFrame(index = K_bar, columns = ['ES'])
    dfMES = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    dfKimacro = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    dfPD = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    ECost = pd.DataFrame(index = K_bar, columns = ['Sys'])
    dfPDsys = pd.DataFrame(index = K_bar, columns = ['Sys'])
    
    
    for jK, k_bar in enumerate(K_bar):
        paramsDict['k_bar'] = k_bar
        myPD = PDmodel('min ES', paramsDict, True, True, k_micro)
        dfES.loc[k_bar] = myPD.dict['ESopt']
        dfMES.loc[k_bar] = myPD.dict['MESopt']
        dfKimacro.loc[k_bar] = myPD.dict['k_macro_str']
        dfPD.loc[k_bar] = myPD.dict['PD']
        dfPDsys.loc[k_bar] = myPD.dict['PDsys']
        
            
    'Expected Cost Function'
    Lambda = .18
    Eta = .024
        
    dfFirstTerm = dfPDsys*Lambda*dfES.values
    SCB = Eta*(pd.DataFrame(index = K_bar, data = K_bar, columns= ['Sys']).values - .07)
    dfSecond = (1-dfPDsys)*SCB
    
    ECost = dfFirstTerm  + dfSecond
    
    nMin = np.where(ECost ==  ECost.min())[0]
    print(ECost.iloc[nMin])
    return nMin, ECost, dfPDsys, dfES, SCB


'Specify k_bar grid, and initialize'
k_micro_grid = np.linspace(.05,.11,5)
k_bar_opt = np.zeros_like(k_micro_grid)
Ecost_opt_opt = np.zeros_like(k_micro_grid)


'evaluate each k_bar grid point'
for jKm, k_micro in enumerate(k_micro_grid):
    nMin, ECost, dfPDsys, dfES, SCB = getECostKmicro(k_micro)
    k_bar[jKm] = ECost.iloc[nMin].index[0]
    Ecost_opt_opt[jKm] = ECost.iloc[nMin].values[0]
    print(k_bar)
    ECost.plot()

'plot'
plt.plot(k_micro_grid, Ecost_opt_opt)
