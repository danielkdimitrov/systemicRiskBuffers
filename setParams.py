# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:48:04 2021

@author: danie
"""

import pandas as pd
import numpy as np
import datetime

class SetParams:
        def __init__(self,evalDate='2009-12-28'):
            '''
            For a backtest, this includes also the first date of the first time window
            '''
            self.firstDateBT = pd.to_datetime('2005-01-04') #2016   # 3/31/2015 2006-01-03 2016-01-01 - 2010-01-01 pd.to_datetime('2021-02-15 00:00:00')   # latest
            self.lastDateBT = pd.to_datetime('2022-08-29')    #2021-11-15  pd.to_datetime('2010-01-01 00:00:00') # earliest

            'First Date and Last Date of the analysis.'

            self.firstDate ='2019-08-31'   # 2019-09-09- 2010-01-01 pd.to_datetime('2021-02-15 00:00:00')   # latest
            self.lastDate = '2022-08-29'    # 2021-09-13 2021-11-15 check   pd.to_datetime('2010-01-01 00:00:00') # earliest
            self.universeEURO = ['ABN', 'BAY', 'BBVA', 'BNP', 'CAIX', 'COMZ', 'CRAG', 'CRMU', 'DANK',
                   'DB', 'DZ', 'ERST', 'SWEN', 'HESLN', 'INGB', 'INTE', 'KBCB', 'LBBW',
                   'NORD', 'RABO', 'SAB', 'SANT', 'SEB', 'SOCG', 'SWED', 'UNIC', 'VB'] #self.DataSet.banks[(self.DataSet.banks['Sample']=='Y')| (self.DataSet.banks['Bank Name'] =='Volksbank')].index

            self.tw = 104 #250 # this is only used in the backtest
            self.dt = datetime.timedelta(weeks=self.tw)
            'Modelling parameters'
            self.q = .95
            self.nF = 3  # number of factors
            #self.RR = .0 # recovery rate
            #'Collateral Params'
            #self.mu = 1.
            #self.sigmaC = .1
            'Simulation Parameters'
            self.nSims = 5*10**4 
            #self.df = 5
            '''
            'Universe :'
            if evalDate > pd.to_datetime('2010-07-05')+self.dt:
                self.universe = ['ABN','Rabo','NIBC','Volksbank','NN'] #'ABN','NN' #Aegon                
            else:
                self.universe = ['ABN', 'ING Bank','Rabo','NIBC','Volksbank','NN','Aegon'] #'ABN','NN' #Aegon
            # after 2009-12-04
            self.universeAft2009 = ['ABN', 'Rabo','NIBC','Volksbank','Aegon'] # ex ING, ex NN // after 
            self.universеAft2011 = ['ABN', 'ING Bank','Rabo','NIBC','Volksbank','Aegon'] # ex NN
                
            '''
            #self.universeFull = ['ABN', 'INGB','RABO','NIBC','VB','AEGO', 'NN'] 
            
            #Nf = len(self.universe)
            
            '''
            'The individual dates for each firm'
            startDates = ['2010-01-01'] #'2016-11-14','2016-11-20'
            endDates = ['2021-02-15']#'2021-02-15','2021-02-15'
            
            self.startDates, self.endDates = [0]*Nf, [0]*Nf
            for jD, (stD, endD) in enumerate(zip(startDates,endDates)):
                self.startDates[jD] = pd.to_datetime(stD + ' 00:00:00')
                self.endDates[jD] = pd.to_datetime(endD + ' 00:00:00')
          '''
            'Merton :'
            self.r = 0.005 
            self.T = 1. 
            'Data frequency'
            self.dt = 1/52
            
            'sytem'
            self.path = 'C:/Users/danie/Dropbox/Amsterdam/CoVaR_Project/images/' #the path for saving figures


''' 
mask = DataSet.CDSprices['Aegon'].notna()
DataSet.CDSprices[mask].tail()

ING Bank : 
2010-07-05
Aegon : 
2010-06-28

all the others : 
Rabo 
2009-12-28
''' 