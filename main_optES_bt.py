# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:15:30 2023

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from optimalSystemicCapital import PDmodel
from myplotstyle import * 
#from GetSystemicRiskSims import *
from GetImpliedParamsBT import GetImpliedParams
from SystemicRisk import FactorModel


from adjustText import adjust_text
from datetime import timedelta
from scipy.stats import norm

#from scipy.optimize import root, minimize, Bounds
#from statsmodels.stats.correlation_tools import cov_nearest

import pickle

path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\imagesTemp\\"


#%%% Backtest

'''
-- run as backtest  
'''


Params = SetParams()
universe = Params.universeEURO
tw = Params.tw

'get data'
myImpliedParams = GetImpliedParams()

dfCDS = myImpliedParams.dfCDS #CDSs processed and in pc form


'Initialize'
paramsDict = {}
paramsDict['Lbar'] = .0
paramsDict['k_bar'] = 0.14 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myImpliedParams.DataSet.banks.loc['DANK', 'p2r  CET1'] = myImpliedParams.DataSet.banks['p2r  CET1'].median()

dfKmacro = pd.DataFrame(index = dfCDS.index[0:5], columns =universe )

'Run backtest'
for indexDate, debtEval in dfCDS.loc[Params.lastDateBT:Params.firstDateBT].iterrows():
    'carve out the data from the time window'        
    # add here universe - carveout 
    twStDate = dfCDS.loc[indexDate].name - timedelta(weeks=tw)              
    if twStDate < Params.firstDateBT:
        break

    'Mask the time window for the current rolling window selection'
    mask = (dfCDS.index > twStDate) & (dfCDS.index <= indexDate)
    'mask away companies with enough data. The rest to be dropped from current sample'
    dfNAs = myImpliedParams.dfU[mask].isna().sum()
    currentBanks = dfNAs[dfNAs<1].index    

    print('\ncalculating date :', indexDate)
    print('\t exclude :',dfNAs[dfNAs>15].index.tolist() )
    #print(myImpliedParams.wts[mask].loc[indexDate].sum())

    'Get PDs on a rolling basis'
    ldngs = FactorModel(myImpliedParams.dfU[mask][currentBanks], Params.nF).ldngs 
    
    paramsDict['Sigma'] = myImpliedParams.dfSigma[mask].loc[indexDate,currentBanks].values.astype(np.float64)
    paramsDict['wts'] = myImpliedParams.wts[mask].loc[indexDate,currentBanks].values.astype(np.float64)
    paramsDict['LGD'] = myImpliedParams.LGD[mask].loc[indexDate,currentBanks].values.astype(np.float64)
    paramsDict['Rho'] = ldngs
    paramsDict['Names'] = currentBanks
    paramsDict['k_p2r'] = myImpliedParams.DataSet.banks['p2r  CET1'][currentBanks].values.astype(np.float64)
    #paramsDict['O-SII rates'] = myImpliedParams.DataSet.banks.loc[currentBanks,'O-SII buffer'].values.T[0]


    myPD = PDmodel('min ES', paramsDict)

    'keep output in a df' 
    dfKmacro.loc[indexDate, currentBanks] = myPD.dict['k_macro_str']*100
    dfKmacro.to_excel('dfKmacro_bt.xlsx')
    #myPD.dict['k_str'] 
    
    
dfKmacro = pd.read_excel('dfKmacro_bt.xlsx', index_col='Date')


'Filter out spikes'
filtered_data = dfKmacro.copy()
spike_threshold = .005
percentage_change = dfKmacro.pct_change()

filtered_data[abs(percentage_change) > spike_threshold] = None

temp = filtered_data.head(1).T
labelsList = temp.sort_values(by=temp.columns[0], ascending=False).index
labelsList = labelsList[:9]

# Interpolate missing values
filtered_data.interpolate(method='linear', inplace=True)


plt.figure(figsize=(12, 6))
# Plot the filtered and interpolated data
ls = ['solid','dotted','dashed','dashed','dashdot',(0, (1, 10)), (0, (1, 1))]*10
i = 0
lines = []
for column in filtered_data.columns:
    line, = plt.plot(filtered_data.index, filtered_data[column], alpha=0.7, linestyle = ls[i])
    lines.append(line)
    i = i +1

# Annotate each line with its label
annotations = []

for line, column in zip(lines, filtered_data.columns):
    x_pos = filtered_data.index[-1]
    y_pos = filtered_data[column].iloc[-1]
    label = column
    myColor = line.get_color()
    if label in labelsList: 
        myAnnotation = plt.text(x_pos, y_pos, label, fontsize=12, color = myColor)
        annotations.append(myAnnotation)

# Adjust the label positions to prevent overlap
adjust_text(annotations, expand_text=(1.02, 1.02))
plt.xticks(fontsize=20)  # Adjust font size here
plt.yticks(fontsize=20)  # Adjust font size here
saveFig(path,'ES_bt') 
plt.show()

#plot also weights and compare, do they dominate 


