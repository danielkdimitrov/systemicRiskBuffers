# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:16:23 2022

@author: danie
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:41:39 2021

@author: danie


This script runs 

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats
import seaborn as sn
from scipy import optimize


from setParams import SetParams
from DataLoad import *
from SystemicRisk import *
from myplotstyle import * 

Params = SetParams()

class GetImpliedParams:
    def __init__(self, dutchSubsample = True, fixedRRs = True, plotInd = False, evalDate = Params.lastDate, startDate=Params.firstDate, RR = .2):
        '''
        Parameters
        ----------
        varyParam : Bool
            Do plots True/ False.
        plotInd : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        
        -------
        I need to fix other DataSet = DataTransform(getEquity = False)

        '''
        self.fixedRRs = fixedRRs
        self.dutchSubsample = dutchSubsample
        self.plotInd = plotInd
        
        
        self.nF = 3 #3 # number of factors        
        self.DataSet = DataTransform(getEquity = False)
        'Set Universe and Time Window'
        self.Params = SetParams()    
        self.evalDate, self.startDate = evalDate, startDate  #self.Params.lastDate, self.Params.firstDate #'2021-09-13','2019-09-09'  #2016-12-07 #2010-01-01  '2021-11-15',
        'Set the Universe'        
        self.universe = Params.universeEURO #self.DataSet.banks[(self.DataSet.banks['Sample']=='Y')| (self.DataSet.banks['Bank Name'] =='Volksbank')].index #  | (self.DataSet.banks['Bank Name'] =='NIBC Bank')
        
        self.path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\images2\\"
        self.mycolors = ['k', 'tab:blue', 'tab:grey', 'tab:orange', 'tab:brown', 'tab:green', 'tab:pink', 'tab:olive']   
        self.myLineStyle = ['-','--','-.',':','^-']   
        'mapping Code : Name ' #  self.DataSet.banks['Bank Name'].loc[universe]
                
        
        '----- Process Data -----'
        # Ensure the DataFrame index is sorted
        self.DataSet.debt = self.DataSet.debt.sort_index()                
        Debt = self.DataSet.debt.loc[self.startDate:self.evalDate,self.universe]
        self.debtEval = self.DataSet.debt[self.universe].loc[self.evalDate].squeeze(axis=0) #.loc[self.evalDate] #.copy()
        self.debtEval['Sys'] = self.debtEval[self.universe].sum()
        self.wtsEval = self.debtEval.div(self.debtEval['Sys'])
        self.wtsEval = self.wtsEval[self.universe]
        self.wts = Debt.div(self.debtEval['Sys'])
        self.rwa_intensityEval = self.DataSet.rwa_intensity[self.universe].loc[self.evalDate]
        
        self.dfk = self.DataSet.capitalRatio[self.universe]/100
        '0. Get CDS and RRs from the Dataset'
        if self.fixedRRs == True:
            self.dfRR = pd.DataFrame(np.tile(RR,Debt.shape), index= Debt.index, columns = Debt.columns)
        else:         
            self.dfRR = self.DataSet.RR.loc[self.startDate:self.evalDate]
            
        self.dfCDS  = self.getCDSdf()            
        self.LGDEval = 1-self.dfRR[self.universe].loc[self.evalDate].squeeze(axis=0) 
        '1. Get PD, DD data'
        pdd = PD(self.DataSet.rwa_intensity[self.universe], self.dfCDS,self.dfRR, self.dfk, False) # last parameter is True for calculating implied vola
        self.dfPD = pdd.dfPD        
        self.dfU = self.getLatentVar(pdd.dfDD)
        dfCDSeval = pd.DataFrame(self.dfCDS.loc[self.evalDate]).T
        dfRReval = pd.DataFrame(self.dfRR.loc[self.evalDate]).T
        pdd_getSigma = PD(self.DataSet.rwa_intensity[self.universe], dfCDSeval,dfRReval, self.dfk, True) #get impliedVola for evalDate
        self.dfSigmaEval = pdd_getSigma.dfSigma #self.dfSigma.loc[self.evalDate].squeeze(axis=0) #another idea to take the average or ewma mean over the period 
        
        ############################################
        '2. Get Factor Model'
        fm = FactorModel(self.dfU, self.nF)
        self.ldngs = fm.ldngs
        

    def getCDSdf(self):
        
        '0. Add premium on banks with SR CDS rate'
        dfCDSraw = self.DataSet.CDSprices.loc[self.startDate:self.evalDate, self.universe]
        
        'Adjust data'
        SUBmask = (self.DataSet.banks['CDS']=='SUB') & (self.DataSet.banks['Sample']=='Y') & (~self.DataSet.banks['Country'].isin(['Spain','Italy']) )
        SRmask = (self.DataSet.banks['CDS']=='SR') & (self.DataSet.banks['Sample']=='Y')
        
        SR = self.DataSet.banks[['Bank Name']][SRmask].index
        SUB = self.DataSet.banks[['Bank Name']][SUBmask].index
        
        
        dfSpreadSR_Sub = dfCDSraw[SUB].median(axis=1) - dfCDSraw[SR].median(axis=1)

        dfCDSadj = dfCDSraw.copy()
        
        for cloumn in dfCDSadj[SR]:
            dfCDSadj[cloumn] = dfCDSadj[cloumn] + dfSpreadSR_Sub.clip(0,None)

        if self.plotInd == True:    
            dfCDSadj[SR].plot()
        
        '0. Get CDS data in proper form' 
        dfCDS = dfCDSadj/1e4
        dfCDS.interpolate(method='quadratic', inplace=True)
        #dfCDS.plot(subplots=True, title = 'CDS ')
                
        return dfCDS
    
    
    def getLatentVar(self, dfDD):
                
        'Get DD log changes'
        dfU = dfDD.diff(-1)
        #dfU = dfDD.transform(lambda x: np.log(x.astype('float64'))).diff(-1)
        dfU = dfU[:-1] # drop the first NoN obs
                    
        return dfU