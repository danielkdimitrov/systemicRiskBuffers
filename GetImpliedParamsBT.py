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

class GetImpliedParams():
    def __init__(self):
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
        Params = SetParams()
        firstDate = Params.lastDate 
        lastDate=Params.firstDate
        
        self.RR = .2
        self.nF = 3 #3 # number of factors        
        self.DataSet = DataTransform(getEquity = False)
        'Set Universe and Time Window'
        self.firstDate, self.lastDate = Params.firstDateBT, Params.lastDateBT  #self.Params.lastDate, self.Params.firstDate #'2021-09-13','2019-09-09'  #2016-12-07 #2010-01-01  '2021-11-15',
        'Set the Universe'        
        self.universe = Params.universeEURO #self.DataSet.banks[(self.DataSet.banks['Sample']=='Y')| (self.DataSet.banks['Bank Name'] =='Volksbank')].index #  | (self.DataSet.banks['Bank Name'] =='NIBC Bank')
                        
        
        '----- Process Data -----'        
        Debt = self.DataSet.debt.loc[self.lastDate:self.firstDate,self.universe]
        self.wts = Debt.div( Debt.sum(axis='columns'), axis='index' )
        
        self.dfk = self.DataSet.capitalRatio.loc[self.lastDate:self.firstDate,self.universe]/100
        '0. Get CDS and RRs from the Dataset'
        self.dfRR = pd.DataFrame(np.tile(self.RR,Debt.shape), index= Debt.index, columns = Debt.columns)
            
        self.dfCDS  = self.getCDSdf()            
        self.LGD = 1-self.dfRR
        '1. Get PD, DD data'
        pdd = PD(self.dfCDS,self.dfRR, self.dfk, True) # last parameter is True for calculating implied vola
        self.dfPD = pdd.dfPD
        self.dfSigma = pdd.dfSigma
        
        self.wts = self.wts
        
        ############################################
        '2. Get Factor Model'
        self.dfU = self.getLatentVar(pdd.dfDD)        
        #fm = FactorModel(dfU, self.nF)
        #self.ldngs = fm.ldngs
        

    def getCDSdf(self):
        
        '0. Add premium on banks with SR CDS rate'
        dfCDSraw = self.DataSet.CDSprices.loc[self.lastDate:self.firstDate, self.universe]
        
        'Adjust data'
        SUBmask = (self.DataSet.banks['CDS']=='SUB') & (self.DataSet.banks['Sample']=='Y') & (~self.DataSet.banks['Country'].isin(['Spain','Italy']) )
        SRmask = (self.DataSet.banks['CDS']=='SR') & (self.DataSet.banks['Sample']=='Y')
        
        SR = self.DataSet.banks[['Bank Name']][SRmask].index
        SUB = self.DataSet.banks[['Bank Name']][SUBmask].index
        
        
        dfSpreadSR_Sub = dfCDSraw[SUB].median(axis=1) - dfCDSraw[SR].median(axis=1)

        dfCDSadj = dfCDSraw.copy()
        
        for cloumn in dfCDSadj[SR]:
            dfCDSadj[cloumn] = dfCDSadj[cloumn] + dfSpreadSR_Sub.clip(0,None)
        
        '0. Get CDS data in proper form' 
        dfCDS = dfCDSadj/1e4
        dfCDS.interpolate(method='quadratic', inplace=True)
                
        return dfCDS
    
    
    def getLatentVar(self, dfDD):
                
        'Get DD log changes'
        dfU = dfDD.diff(-1)
        #dfU = dfDD.transform(lambda x: np.log(x.astype('float64'))).diff(-1)
        #dfU = dfU[:-1] # drop the first NoN obs        
            
        return dfU