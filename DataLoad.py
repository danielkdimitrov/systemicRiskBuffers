"""
Created on Thu Nov  5 17:35:43 2020

Load and plot the data

@author: danie
"""
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy import optimize

import statsmodels.formula.api as smf

from setParams import SetParams

class DataLoad(SetParams):
    def __init__(self):
        'Loading up the data'
        #'NN : too short of a time history - add later'
        #self.universe = SetParams().universeFull # ['Aegon','ABN','NN'] # CDS : ['ABNsub','ING','Rabo','NIBC','Volksbank', 'AEGON'] 
        #  self.setConstants()
        'TODO : Put this with the Merton Class'
        
    def getCDS(self):
        cdsPrice = self.loadAllFiles('data\cds','w')
        cdsPrice = cdsPrice #in decimals 
        cdsLC = self.getLogChanges(cdsPrice)
        return cdsPrice, cdsLC

    def getEquity(self):
        equityPrice = self.loadAllFiles('data\equity','w')
        equityLC = self.getLogChanges(equityPrice)
        return equityLC, equityPrice

    def getEquityMV(self):
        return self.loadAllFiles('data\equityMV','w')
    
    def getDebt(self):
        return self.loadAllFiles('data\debt','a')
    
    def getTier1Ratio(self):
        return self.loadAllFiles('data\\tier1capratio','a')
    
    def getVola(self):
        return self.loadAllFiles('data\\vola','w')   
        
    def loadAllFiles(self,subFolder,freq):
        '''
        folder : string of the child folder where the csv files are
        '''
        df = pd.DataFrame()
        for file_name in glob.glob(subFolder+'\*.csv'):
            print('Loading Data: ', file_name)
            series = pd.read_csv(file_name, header=0, parse_dates=["Date"], index_col=0, squeeze=True)
            if df.empty == False:
                df = pd.merge(df,pd.DataFrame(series) , on='Date', how='outer')
            else: 
                df = pd.DataFrame(series)
        if freq == 'w':
            'Keep Weekly data only : '
            #freq = freq # now, either 'w' for weekly or otherwise it's daily by default
            df = self.getWeekly(df)
        'TODO : if a (which is debt) fill in missing values to daily/weekly'
        'sort so that newest ? are on top : '
        df.sort_index(ascending=False, inplace=True)
        # else keep as it is
        return df
    
    def getWeekly(self,df):
        'Resample the daily Price data to weekly. Resmapling is done to keep Monday prices '
        #offset = pd.offsets.timedelta(days=-6)
        df = df.resample('W', label='left', loffset=pd.DateOffset(days=1)).first() #.first() #,loffset=offset
        df.sort_index(ascending=False, inplace=True)
        
        return df
            
    def getLogChanges(self, dfRaw):
        df = pd.DataFrame(index=dfRaw.index)
        
        'Get Log Returns'
        df = dfRaw.transform(lambda x: np.log(x)).diff(-1)
        'define the system. TODO : I need to be taking weights into account later on'
        df['Sys'] = df.mean(numeric_only=True, axis=1) #[self.universe]
                
        return df

class DataTransform(DataLoad):
    def __init__(self, getEquity=False):
        'Start to End data for the Backtests'
        self.endDate =  SetParams().firstDateBT #indxDates.max()
        self.startDate = SetParams().lastDateBT  #indxDates.min() 
    
        'Load : 1/ CDS price, and log-changes or Market Value'
        self.CDSprices, self.CDSreturns = DataLoad().getCDS()
        
        if getEquity == True:
            'Load : 2/ Market Value and Equity Returns'        
            self.eqMV = DataLoad().getEquityMV()
            self.eqReturns, self.eqPrices = DataLoad().getEquity()
            self.eqStd = self.eqReturns.sort_index(ascending=True).rolling(52).std()*np.sqrt(52) #250 50 weeks rolling, scale to a year
            self.eqStd.sort_index(ascending=True, inplace=True)            
         
        'Load : 3/ Debt, Recovery Rate, Deposits to Liabs Ratio'        
        debt = DataLoad().getDebt()
        self.debt = self.getToWeekly(debt)
        'Load Capital 1 Ratios'
        capitalRatio = DataLoad().getTier1Ratio()
        self.capitalRatio = self.getToWeekly(capitalRatio)
        print('Loading Data: data\\other\RR.csv; DepositsToLiabs.csv')        
        RR = pd.read_csv('data\\other\\RR.csv', header=0, parse_dates=["Date"], index_col=0, squeeze=True)
        self.RR = self.getToWeekly(RR)
        DL = pd.read_csv('data\\other\\DepositsToLiabs.csv', header=0, parse_dates=["Date"], index_col=0, squeeze=True)
        self.DL = self.getToWeekly(DL)
        self.banks = pd.read_csv('data\\other\\BankDefinitions.csv', header=0,index_col=0, squeeze=True)
        'Load Capital Ratios'

        'Get : Rolling StDev./Weekly'
        self.CDSstd = self.CDSreturns.sort_index(ascending=True).rolling(52).std()*np.sqrt(52) #250 50 weeks rolling, scale to a year
        self.CDSstd.sort_index(ascending=True, inplace=True)
        self.vola = DataLoad().getVola()/100
        
    def getToWeekly(self, df):
        '- Create empty row'
        df = df.append(pd.Series(name=self.endDate)) #
        df.sort_index(ascending=False, inplace=True)
        '- Get down to weekly and fill in missing'
        df = DataLoad().getWeekly(df)
        df = df.interpolate(method='quadratic')  #bfill() #    # fill backward with the same value as the most recent forward
        df = df.bfill() #
        return df


'Some data loading and plotting examples follow below'
   
#DataSet = DataTransform()

'''
MM = DataTransform() #
dfEq = MM.eqMV


MM = MertonFirmModel('Equity')
dfVeq = MM.merton.xs('V')


names = ['Aegon', 'NN','ABN']

'Plot joing Vola'
fig, ax = plt.subplots(nrows=3, ncols=1)
for jN, name in enumerate(names):
    MM.CDSstd[name].plot(ax = ax[jN], label='Sigma(CDS)')
    MM.eqStd[name].plot(ax = ax[jN], label='Sigma(Eq)')
    ax[jN].legend()
    ax[jN].set_title(name)

fig2, ax2 = plt.subplots(nrows=3, ncols=1)
#ax20 = ax2.twinx()
for jN, name in enumerate(names):
    MM.CDSprices[name].plot(ax = ax2[jN], label='CDS')
    ax20 = ax2[jN].twinx()
    MM.eqPrices[name].plot(ax = ax20, label='EQ Price', color='tab:orange')   
    ax2[jN].set_ylabel('CDS (blue)')
    ax20.set_ylabel('EQ Price (red)')
    ax2[jN].set_title(name)
''' 


'''

#dfDebtInterpol = MM.debt
dfDebt = MM.debt
'''
'''
ax = dfDebtInterpol['Aegon'].plot()
dfDebt['Aegon'].plot(ax=ax)
plt.title('Debt Interpolation, Aegon')
plt.savefig(path+'debtIntrplAegon.pdf',bbox_inches = 'tight') #sigma_x
'''

'''
'Plotin a loop'
figs1 = [0,0,0,0,1,1,1,1]
figs2 = [0,1,2,3,0,1,2,3]

fig, ax = plt.subplots(8)
for jN, Name in enumerate(SetParams().universe):
    ax[figs1[jN],figs2[jN]] = dfDebt[Name].plot()
    ax[figs1[jN],figs2[jN]] = dfDebtInterpol[Name].plot()
    
    #dfDebt[Name]['V'].plot(1)
    #dfDebt[Name]['V'].plot(1)
    #MM.debt[Name].plot()
    #MM.stdE[Name].plot()
    #MM.merton[Name]['V'].plot()
plt.legend()
'''

''' 

MM.prices.plot(subplots=True) #sharey=True, ,title='CDS Prices (bps)'
plt.savefig(path+'CDSprices.pdf',bbox_inches = 'tight') #sigma_x


MM.debt.plot(subplots=True) #sharey=True,,title='Debt'
plt.savefig(path+'debt.pdf',bbox_inches = 'tight') #sigma_x
'''

''' Plots

MM.merton.xs('V').plot()
MM.prices['Aegon'].plot()

MM.merton.xs('sigmaV').plot()
'''

###############

'''
'Save to excel'
MM.merton.xs('V').to_excel('V_CDS.xlsx')
MM.debt['Aegon'].to_excel('V_dbt.xlsx')

'Plotin a loop'
for jN, Name in enumerate(SetParams().universe):
    #df[Name]['V'].plot(1)
    #MM.debt[Name].plot()
    #MM.stdE[Name].plot()
    #MM.merton[Name]['V'].plot()
plt.legend()

MM.prices.plot()

MM.debt[SetParams().endDate:SetParams().startDate][SetParams().universe].plot()
'''
###############
'''
dfEQ = DataLoad().getEquityMV()

dfEQ['Aegon'].plot()


dfDebt = DataLoad().getDebt()

dfDebt['Aegon'].plot()
'''

'''     
dfRaw = MyData()

dfEMV = dfRaw.equityMV
dfDebt = dfRaw.debt
dfStd = dfRaw.stdE
dfEq = dfRaw.equity
''' 
