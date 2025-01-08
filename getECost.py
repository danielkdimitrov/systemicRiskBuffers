# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:44:14 2023

@author: danie
"""
import pandas as pd
import numpy as np

from optimalSystemicCapital import PDmodel 


def getECost(paramsDict):

    
    K_bar = np.linspace(0,.2,30) #  .15 // 0.05,.1
    LGD = paramsDict['LGD']
    dfES = pd.DataFrame(index = K_bar, columns = ['ES'])
    #dfMES = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    dfKimacro = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    #dfPD = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
    ECost = pd.DataFrame(index = K_bar, columns = ['Sys'])
    dfPDsys = pd.DataFrame(index = K_bar, columns = ['Sys'])
    'collect Lsys simulations in a dictionary :'
    dict_Lsys = dict.fromkeys(K_bar)
    
    'Loop through the different k-bars in K_bar grid'
    for jK, k_bar in enumerate(K_bar):
                    
        paramsDict['k_bar'] = k_bar
        myPD = PDmodel('min ES', paramsDict, True, True)
        dfES.loc[k_bar] = myPD.dict['ESopt']
        #dfMES.loc[k_bar] = myPD.dict['MESopt']
        dfKimacro.loc[k_bar] = myPD.dict['k_macro_str']
        #dfPD.loc[k_bar] = myPD.dict['PD']
        dfPDsys.loc[k_bar] = myPD.dict['PDsys']
        
        dict_Lsys[k_bar] = myPD.Lsys
        
        if k_bar == 0:
            Lambda = paramsDict['GDP Loss']/myPD.dict['ESopt']
            print('Lambda=', Lambda)

                        
    #Lambda = .09#/(Lbar*np.sum(LGD*paramsDict['wts'])) #.18 for LGD = 100% , evaluate at the weighed mean LGD
    Eta = .024

    'Expected Cost Function'
    SCB = Eta*(pd.DataFrame(index = K_bar, data = K_bar, columns= ['Sys']).values)#.07
        
    dfFirstTerm = dfPDsys*Lambda*dfES.values
    dfSecond = (1-dfPDsys)*SCB
    
    ECost = dfFirstTerm  + dfSecond
    
    nMin = np.where(ECost ==  ECost.min())[0]
    k_bar_min = ECost.iloc[nMin]

    myDict = {'ECost': ECost, 'k_bar_min': k_bar_min, 'dfPDsys': dfPDsys, 'dfES': dfES, 'SCB': SCB, 'Lambda': Lambda,'K_bar':K_bar, 'Lsys':dict_Lsys}

    return myDict
