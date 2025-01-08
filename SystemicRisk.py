# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:29:32 2021

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from datetime import timedelta
from optimalSystemicCapital import PDmodel

from scipy.stats import norm, beta, triang
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest
from scipy import optimize
#from sklearn.covariance import LedoitWolf

class PD:
    def __init__(self, dfCDSp, dfRR, dfk, getVola = False):
        '''
        Get Implied Probab of Default, and Distance to Default from CDS prices
        INPUT : 
            dfCDSp : dataframe - CDS prices : divide first by 1e4
            dfk : dataframe - tier 1 capital ratio
        OUTPUT : 
            dfPD : dataframe - Implied (from CDS rates) Probability of Default
            dfDD : dataframe - Implied Distance to Default 
            dfSigma : dataframe - Implied RWA variance based on Tier 1 CR
        '''
        
        'initialize'
        self.r= .005
        
        self.dfPD = pd.DataFrame(index=dfCDSp.index, columns = dfCDSp.columns)
        self.dfDD = pd.DataFrame(index=dfCDSp.index, columns = dfCDSp.columns)
        self.dfSigma = pd.DataFrame(index=dfCDSp.index, columns = dfCDSp.columns)
        

        for jN, Name in enumerate(dfCDSp.columns):
            for indexDate, CDS in dfCDSp[Name].items():
                RR = .2 #dfRR.loc[indexDate,Name]
                PD, DD = self.getPD(CDS, 1 - RR)
                self.dfPD.loc[indexDate,Name], self.dfDD.loc[indexDate,Name] = PD, DD
                if getVola == True:
                    k = dfk.loc[indexDate,Name]

                    self.dfSigma.loc[indexDate,Name] = self.getVola(PD,k)

            
    def getPD(self, CDS, LGD):
        '''
        r : risk-free rate 
        T : maturity of the CDS
        dt: descrete time period  
        '''
        r, dt, T  = self.r, 1, 5 #LGD .8
        
        a = (1/r)*(1-np.exp(-T*dt*r) )
        b =  ((1/r)**2) *(1 - (T*dt*r + 1)*np.exp(-T*r*dt))
        'Probability of default:'
        PD = (a*CDS) / (a*LGD + b*CDS)
        'Distance to Default'
        DD = - norm.ppf(PD)
        return PD, DD  


    def DD(self, k, sigma):
        'k, sigma could be arrays with elements standing for each bank'
        DD = (-np.log(1. - k) + (self.r - (1/2) *sigma**2)) / sigma
        PD = norm.cdf(- DD)
        return PD
    
    
    def pd_sigma(self, pd, k, sigma):
        
        PD_merton = self.DD(k,sigma)
        res = pd - PD_merton
        
        return res
    
    def getVola(self, PD, k):
        'minimize the diff btw observed PD and calculated via Merton'
        f = lambda sigma: self.pd_sigma(PD, k, sigma)
        sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
        return sol.root



class FactorModel:
    def __init__(self, dfU, nF=1):
        '''
        Algorithm to get the factor loadings of the model, based on Andersen 2003.
        INPUT : 
            dfU : Input from the PD class. It's the standard normal inverse of the probability of default,
            representing the unstandardized asset reutrns. 
            nF : number of factors, by default 1
        '''
        'standardize, so that the cov matrix later on is the corr matrix'
        dfStd = (dfU -dfU.mean())/dfU.std()
        self.nF = nF  # number of factors 

        'get the loadings'
        self.ldngs, self.Cov = self.getLoadings(dfStd)
        'get the factors'
        self.fctrs = dfStd.dot(self.ldngs)
        

    def getLoadings(self, dfStd):
        '''
        Implementation of the Andersen 2003 Algorithm
        
        INPUT : 
            dfStd - standardized dataFrame containing the variables time series
        
        OUTPUT : 
            c - matrix of factor loadings, sum of squares less than 1
        '''
        def nearPSD(A):
            C = (A + A.T)/2
            eigval, eigvec = np.linalg.eig(C)
            eigval[eigval < 0] = 0
        
            return eigvec.dot(np.diag(eigval)).dot(eigvec.T)
           
        Cov = np.cov(dfStd.T.astype(float))
        n = dfStd.columns.size # number of variables 
        F = np.diag(np.ones(n)*0.001)
        
        eps, iters = 100, 0
        Niters = 2500
        while (eps > 1e-6 and iters < Niters):
            iters += 1 
            SigmaF = Cov - F
            #SigmaF1 = cov_nearest(SigmaF) # to ensure cov matrix is positive definite
            eigenval, eigenvect = np.linalg.eig(SigmaF)
            if np.any(np.iscomplex(eigenval)) == True:
                print('complex egienvalues!')
                print(eigenval)
                break
            # what to do when eigenval is negative ???
            if np.any(eigenval[:self.nF] < 0) == True:
                print('Make SigmaF positive semi-definite')
                SigmaF = nearPSD(SigmaF)
                eigenval, eigenvect = np.linalg.eig(SigmaF)
                #print('New Eigenvalues:', np.all(eigenval) < 0)
                #break
                '''
                SigmaF = Cov - F
                eigenval, eigenvect = np.linalg.eig(SigmaF)
                print(eigenval[:nF])
                #print(np.linalg.norm(cov_nearest(SigmaF) - SigmaF))
                '''
            
            LambdaM =np.diag(eigenval[:self.nF])
            sqLambdaM = np.linalg.cholesky(LambdaM)
            
            E = eigenvect[:,:self.nF]
            c = np.matmul(E,sqLambdaM)
        
            cc = np.matmul(c, c.T)
            Fnew = np.diag(1- np.diag(cc))
            eps = np.linalg.norm(F-Fnew)
            '''np.linalg.norm(eigenvect)
            if iters % 5 == 0:
                print(iters)                
                print('epsilon = ', eps)
            '''
            F = Fnew.copy()
        if iters > Niters: 
            print('Factor model did not converged')
            print('iters: ',iters,'eps: ',eps)            
            print('eps:', eps)
        if c[0,0] < 0:
            c = -c
            print('scaled loadings by -1')
        return c, Cov
    
class LossesSim:
    def __init__(self, sigmaC, ldngs, Debt, dfDD, dfERR, lossType, distrib = 'Normal', printt=False, delta = -1., nu=6.,dependLoss=True, rho=0):
        '''
        INPUT : 
            dfDD - SERIES - distance to default
            fLoadings - NUMPY ARRAY - factor loadings 
            DD - SERIES - distance to default
            dependLoss, rho: if False, assuming one-factor for the collateral (LGDs), provide rho factor coefficient 
        OUTPUT : 
            A matrix of simulated losses
            add : CVaR, CCoVaR
        '''
        universe = dfDD.index
        
        np.random.seed(1)

        'Simulate Factor Model : nF factors  + 1 residual in factor model + 1 residual in collateral'
        #M = sims[:,:nF]
        #dZ = sims[:,nF:-1]
        #dZc = sims[:,-1]
        #np.random.normal(loc=0, scale=1, size = (nSims, nF+2))
                
        'Get simulated defaults as of the evalDate. Make into a Rolling Window'       
        #evalDate = Debt.name   # !!! Check evalDate
        #IndctrD = np.zeros(SetParams().nSims)
        'System debt is the debt in all institutions'
        Debt_Sys = Debt[universe].sum()
        
        Lsim = pd.DataFrame(columns = dfDD.index)
        self.RRsim = pd.DataFrame(columns = dfDD.index) #index=range(nSims),
        self.dWsim = pd.DataFrame(columns = dfDD.index) 
        self.IndctrD = pd.DataFrame(0, index=np.arange(SetParams().nSims), columns = dfDD.index) 
        
        self.ELGD = pd.DataFrame(index=[0], columns = dfDD.index)
        #np.zeros([DD.columns.size,nSims])
        #self.TC = triang.rvs(.0107,loc=.12,scale = .28, size=nSims)
        # if the object is series, turn it into df 
        nA, nF = ldngs.shape # number of firms and number of factors
        'Get Factor Simulations'
        nSims = SetParams().nSims
        M = np.random.normal(loc=0.0, scale=1.0, size=(nF, nSims))  #Factor
        dZ = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))  # Idiosynch 
        dZc = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))

        if distrib =='stt' or distrib == 'skewedstt':
            'Generate fat-tailed/skewed factor'
            #nu = 6.             
            V = np.random.chisquare(nu, nSims)
            #delta = -1.
            W = np.random.normal(loc=-np.sqrt(2/np.pi), scale=1.0, size=(nSims))
            W[W<-np.sqrt(2/np.pi) ]= 0                
         
        for jN, Name in enumerate(dfDD.index):
            'Simulate Factor Loadings'                        
            A = ldngs[jN,:] # factor loadings for firm jN
            
            dW = M.T@A + np.sqrt(1- A.T@A)*dZ[jN,:]
            
            if distrib =='stt' or distrib == 'skewedstt':
                
                if distrib == 'skewedstt':
                    dW =  np.sqrt(nu/V)*(delta*W + dW)
                else: 
                    dW = np.sqrt(nu/V)*dW
                                        
            'Simulate RR : use only the first factor'
            C =M.T@A + np.sqrt(1- A.T@A)*dZc[jN,:] # this is dWc in the paper #dZc[jN,:]#

            if dependLoss == False:
                A_hat = rho*A/(np.sqrt(A.T@A))
                C = M.T@A_hat + np.sqrt(1- A_hat.T@A_hat)*dZ[jN,:]
                #rho*M[0] + np.sqrt(1- rho**2)*dZc[jN,:] # this is dWc in the paper #dZc[jN,:]#
            
            if distrib == 'skewedstt':
                C =  np.sqrt(nu/V)*(delta*W + C)
            if distrib == 'stt': 
                C= np.sqrt(nu/V)*C
            
            #sigmaC = .15
            #RR = np.minimum(1, np.exp(sigmaC*C) ) #Collateral #SetParams().
            #mu_c = dfERR[Name]/np.mean(RR)
            RR = norm.cdf(C) #mu_c*RR

            if printt == True : print('Simulating:', Name)
            self.IndctrD[Name][dW <= - dfDD.loc[Name]] = 1
            
            Lsim[Name] = self.IndctrD[Name]*Debt.loc[Name]*(1-RR)
            #self.IndctrD[Name] = IndctrD
            self.ELGD.loc[0,Name] = Lsim[Name][self.IndctrD[Name]>0].mean()/Debt[Name] # expected loss given default
            self.RRsim[Name] = RR
            self.dWsim[Name] = dW
        
        'originally Lsim (losses) are in euro'
        Lsim['Sys'] = Lsim.sum(axis=1) #get systemic losses in each scenario
        
        if lossType == 'Sys':
            'Losses as percent of system debt (sum of all firms debt)'
            Lsim = 100*(Lsim.div(Debt_Sys))
            
        if lossType == 'PCD':
            'Losses as percent of own banks debt'
            Lsim[universe] = 100*Lsim[universe].div(Debt[universe])
            Lsim['Sys'] = 100*Lsim['Sys'].div(Debt_Sys)
        
        self.IndctrD['Sys'] = self.IndctrD.sum(axis=1) #number of defaults        
        self.M = pd.DataFrame(M.T)
        self.Lsim = Lsim        
        self.Debt = Debt

class CoVaR:
    def __init__(self,dfLsim):
        '''
        INPUT : 
            dfLsim - dataframe - simulated Losses
            q - the tail probability. Used to calculate VaR(1-q), CoVaR(1-q)

        '''
                
        q = SetParams().q

        VaR = pd.DataFrame(index = [q, .99, .5], columns = dfLsim.columns)
        CoVaR = pd.DataFrame(index = [q, .99, .5], columns = dfLsim.columns)
        self.ECoVaR = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)
        self.MES = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)  
        self.ES = pd.DataFrame(index = [q, .99,  .5], columns = dfLsim.columns)
        self.ExSq = pd.DataFrame(index = dfLsim.columns, columns = dfLsim.columns) #Exposure Shortfall
        self.ExS99 = pd.DataFrame(index = dfLsim.columns, columns = dfLsim.columns) #Exposure Shortfall        

        VaR.loc[q] = dfLsim.quantile(q)
        VaR.loc[.5] = dfLsim.quantile(.5)
        VaR.loc[.99] = dfLsim.quantile(.99)
        
        self.Exptn = dfLsim.mean()
        
        self.MES.loc[q] = dfLsim[dfLsim['Sys']>=VaR.loc[q,'Sys']].mean()
        self.MES.loc[.99] = dfLsim[dfLsim['Sys']>=VaR.loc[.99,'Sys']].mean()
        self.MES.loc[.5] = dfLsim[dfLsim['Sys']>=VaR.loc[.5,'Sys']].mean()        
  
        #self.ELGD.name = Debt.name
        'Calculate VaR, CoVaR, etc. for each entity'
        #dfSys = dfLsim['Sys']
        for jN, Name in enumerate(dfLsim.columns):
            dfLsimCoVaRq = dfLsim[dfLsim[Name]>=VaR.loc[q,Name]]
            CoVaR.loc[q,Name] = dfLsimCoVaRq.quantile(q)['Sys']
            dfLsimCoVaR5 = dfLsim[dfLsim[Name]>=VaR.loc[.5,Name]]            
            CoVaR.loc[.5,Name] = dfLsimCoVaR5.quantile(.5)['Sys']
            dfLsimCoVaR99 = dfLsim[dfLsim[Name]>=VaR.loc[.99,Name]]                        
            CoVaR.loc[.99,Name] = dfLsimCoVaR99.quantile(.99)['Sys']
            
            self.ES.loc[q,Name] = dfLsim[dfLsim[Name]>=VaR.loc[q,Name]][Name].mean()
            self.ES.loc[.99,Name] = dfLsim[dfLsim[Name]>=VaR.loc[.99,Name]][Name].mean()
            self.ES.loc[.5,Name] = dfLsim[dfLsim[Name]>=VaR.loc[.5,Name]][Name].mean()

            self.ExSq[Name] = dfLsim[dfLsim[Name] >= VaR.loc[q,Name]].mean()
            self.ExS99[Name] = dfLsim[dfLsim[Name] >= VaR.loc[.99,Name]].mean()
                                        
        self.ECoVaR.loc[q] = dfLsim[dfLsim['Sys']>=VaR.loc[q,'Sys']].quantile(q)
        self.ECoVaR.loc[.5] = dfLsim[dfLsim['Sys']>=VaR.loc[.5,'Sys']].quantile(.5)
        self.ECoVaR.loc[.99] = dfLsim[dfLsim['Sys']>=VaR.loc[.99,'Sys']].quantile(.99)        
        
        self.VaR = VaR
        self.CoVaR = CoVaR
            
        self.DeltaCoVaR = self.CoVaR.loc[q] - self.CoVaR.loc[.5]
        
class SocialCost:
    def __init__(self,dfCPD, dfPD, dfEAD, dfLGD):
        '''
        INPUT : 
            dfLsim - dataframe - simulated Losses
            wts - dataframe - liability size weights of each institution

        '''
                
        self.df = pd.DataFrame(index = dfCPD.columns, columns =['SCD','Direct Cost', 'Indirect Cost', 'Weight'])
        
  
        #self.ELGD.name = Debt.name
        'Calculate VaR, CoVaR, etc. for each entity'
        #dfSys = dfLsim['Sys']
        for jN, Name in enumerate(dfCPD.columns):
            self.df.loc[Name] =  self.getSCDfromPD(dfCPD, dfPD, dfEAD, dfLGD, Name)
            
                       
    def getSCDfromPD(self, dfCPD, dfPD, dfEAD, dfLGD, Name):
        '''            
        Parameters
        ----------
        dfCPD : dataframe        conditional default probabs vector for all banks conditional on bank j.
        dfPD : dataframe        default probabs for all banks.
        dfEAD : dataframe        debt/liability sizes
        dfLGD : dataframe        loss given default.
        Name : string        reference bank i.
    
        Returns
        -------
        DirectCost : array
        IndirectCost : array
        SCDi : array        social cost of default.
    
        '''
        wts = dfEAD/dfEAD.sum()

        PDi, LGDi, EADi = dfPD.loc[Name].values[0], dfLGD[Name], wts.loc[Name]
        mask_j = ~dfPD.index.isin([Name])
        PDj, LGDj, EADj, PDj_i = dfPD.loc[mask_j].values[:,0], dfLGD.loc[mask_j], wts.loc[mask_j].values, dfCPD.loc[Name][mask_j].values
        
        DirectCost = PDi*LGDi*EADi             
        IndirectCost =np.sum(EADj*LGDj*PDi*(PDj_i -PDj))
        SCDi = IndirectCost + DirectCost
        return SCDi, DirectCost, IndirectCost, wts[Name]
        
            
class DefaultP:
    def __init__(self,IndctrD, getCrossMatrix= False):
        '''
        + p_i_j : Probility of firm i defaulting conditional firm j defaulting
        + p_1, p_2, p_3 : Probability of at least 1, 2 or 3 firms defaulting 
        '''
        #IndctrDSys = IndctrD['Sys']
        NSims = IndctrD.shape[0]  #IndctrDSys.count()
        universe = IndctrD.drop('Sys',axis=1).columns #SetParams().universeFull
        #NFirms = len(universe)
        
        'joint probability of distress : probab. that N or more will default'
        self.p = pd.DataFrame(np.nan, index = ['p1','p2','p3','p4'], columns =['p1ofN', 'pNafter1', 'pNafter2'])
        
        for jP in range(4):
            self.p.iloc[jP,0] = IndctrD['Sys'][IndctrD['Sys']>=jP+1].count() / NSims
            self.p.iloc[jP,1] = self.p.iloc[jP,0] /  self.p.iloc[0,0]
            if jP>1:
                self.p.iloc[jP,2] = self.p.iloc[jP,0] /  self.p.iloc[1,0]
        self.p.iloc[0,1] = np.NAN
        self.p.iloc[0,2] = np.NAN
        self.p.iloc[1,2] = np.NAN            
        'probability that at least one more will default, given that one defaults'
        
                            
        'conditional probab. of distress - loop through each firm conditional on each'
        self.cpd = pd.DataFrame(columns = universe) #expected number of defaults given that a firm defaults
        self.jpd = pd.DataFrame(columns = universe) #expected number of defaults given that a firm defaults
        self.sysIndex = pd.DataFrame(index = universe, columns =['POA','VI','SII'])            

        for jN, Name1 in enumerate(universe):
            'get systemic indicators based on PDs'                            
            CIndctrD = IndctrD[IndctrD[Name1]==1]            
            #dfts = IndctrD['Sys'].value_counts(normalize=True)
            lossScenarios = sum(IndctrD[Name1]==1)
            if lossScenarios > 0:
                self.sysIndex.loc[Name1,'POA'] = 100*sum(IndctrD['Sys'][IndctrD[Name1]==1]>1)/lossScenarios
                self.sysIndex.loc[Name1,'SII'] = IndctrD['Sys'][IndctrD[Name1]>0].mean()
                self.sysIndex.loc[Name1,'VI'] = 100*IndctrD[Name1][IndctrD['Sys']>1].sum()/sum(IndctrD['Sys']>1)
                                    
            for jK, Name in enumerate(universe):
                if getCrossMatrix == True:
                    'Get the Defautl matrix :  All Names conditional on Name1 in distress'
                    for kN, Name2 in enumerate(universe):
                        'probab. that Name2 will be in distress conditional on Name1 in distress'
                        self.cpd.loc[Name1, Name2] = CIndctrD[Name2].sum()/CIndctrD[Name1].count()
                        'joint probability of distress of Name1 and Name2'
                        self.jpd.loc[Name1, Name2] = CIndctrD[Name2].sum()/NSims
        
class RollingWindowCoVaR:
    def __init__(self,DataSet, universe, lossType='PCD', printt=False, distrib='Normal',dependLoss=True):
        '''
        '''
        'Parameters'
        Params = SetParams()
        
        firstDate = Params.firstDateBT 
        lastDate = Params.lastDateBT
        tw = Params.tw
        universeFull = universe
        universeFullSys = np.append(universeFull,'Sys')  # the full universe of firms
        
        '1. Get PD, DD data'
        dfCDS = DataSet.CDSprices.loc[lastDate:firstDate, universeFull]/1e4
        dfCDS.interpolate(method='quadratic', inplace=True)
        #dfRR = DataSet.RR.loc[lastDate:firstDate, universeFull]
        
        dfk = DataSet.capitalRatio/100

        'same RRs assumed'
        RRassumption = .6
        dfRR = pd.DataFrame(np.tile(RRassumption,dfCDS.shape), index= dfCDS.index, columns = dfCDS.columns)
            

        pdd = PD(dfCDS,dfRR, dfk,False) #True
 
        #PDs.dfDD.plot(subplots=True)
        'Get DD log changes'
        dfU = pdd.dfDD.diff(-1)
        #dfU = pdd.dfDD.transform(lambda x: np.log(x.astype('float64'))).diff(-1)        
        #dfU.fillna(0, inplace = True)
                
        dfDebt = DataSet.debt[universeFull].loc[lastDate:firstDate]#.loc[evalDate] #.copy()
        
        dfCDS = DataSet.CDSprices[universeFull].loc[lastDate:firstDate]#.loc[stD:endD]/(10**4)
        self.SigmaC = DataSet.vola.loc[lastDate:firstDate,'vstoxx']
        
        self.VaR = pd.DataFrame(index = dfCDS[lastDate:firstDate + timedelta(weeks=tw)].index, columns = universeFullSys)
        
        #VarList = ['VaR50', 'VaR99', 'CoVaR', 'CoVaR50', 'CoVaR99','ECoVaR','ECoVaR50','ECoVaR99','ES','ES50','ES99','MES','MES50',
        #           'MES99','DeltaCoVaR','ES','POA','VI','SII']
        #df_dict = {name: pd.DataFrame(self.VaR.index,columns = universeFullSys) for name in VarList}
        
        self.VaR50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.VaR99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        
        self.CoVaR = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.CoVaR50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.CoVaR99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        self.ECoVaR = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.ECoVaR50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.ECoVaR99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)

        self.ES = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.ES50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.ES99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        

        self.PCtoES = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.PCtoES50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.PCtoES99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        

        self.MES = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.MES50 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        self.MES99 = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)                
        
        self.DeltaCoVaR = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)
        
        self.ES = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        self.POA = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        self.VI = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        self.SII = pd.DataFrame(index = self.VaR.index, columns = universeFullSys)        
        
        self.LoadingsF1 = pd.DataFrame(index = self.VaR.index, columns = universeFull)
        self.LoadingsF2 = pd.DataFrame(index = self.VaR.index, columns = universeFull)
        self.LoadingsF3 = pd.DataFrame(index = self.VaR.index, columns = universeFull)
        self.FactorShare = pd.DataFrame(index = self.VaR.index, columns = universeFull)                
        self.Weight = pd.DataFrame(index = self.VaR.index, columns = universeFull)
        
        self.pN = pd.DataFrame(index = self.VaR.index, columns = ['p1','p2','p3','p4']) #'p2',
        self.pNafter1 = pd.DataFrame(index = self.VaR.index, columns = ['p1','p2','p3','p4']) #'p2',
        self.pNafter2 = pd.DataFrame(index = self.VaR.index, columns = ['p1','p2','p3','p4']) #'p2',

        'Rolling window'
        for indexDate, debtEval in dfDebt.loc[lastDate:firstDate].iterrows():
            'carve out the data from the time window'
            
            if printt == True: print('\n calculating date :', indexDate)
            # add here universe - carveout 
            twStDate = dfCDS.loc[indexDate].name - timedelta(weeks=tw)              
            if twStDate < firstDate:
                break
            'Mask the time window for the current rolling window selection'
            mask = (dfU.index > twStDate) & (dfU.index <= indexDate)
            'mask away companies with enough data. The rest to be dropped from current sample'
            dfNAs = dfU[mask].isna().sum()
            currentBanks = dfNAs[dfNAs<15].index
            debtEval['Sys'] = debtEval[currentBanks].sum()
            #debtEval = dfDebt.loc[indexDate]
            '2. Get Factor Model'
            dfU_current = dfU[currentBanks].loc[mask].fillna(0)
            #if printt == True: print('missing data, fill in with Zeros :', dfU[currentBanks].loc[mask].isnull().sum())
            fm = FactorModel(dfU_current,Params.nF)
            'Çollect Factor Loadings'
            self.LoadingsF1.loc[indexDate, currentBanks] = fm.ldngs[:,0]
            self.LoadingsF2.loc[indexDate, currentBanks] = fm.ldngs[:,1]
            self.LoadingsF3.loc[indexDate, currentBanks] = fm.ldngs[:,2]
            f_squared = fm.ldngs*fm.ldngs
            self.FactorShare.loc[indexDate, currentBanks] = f_squared.sum(axis=1)             
            
            
            '1. get simulation of the losses. 2. get CoVaR estimates'
            sigmaC = .5 #self.SigmaC.loc[indexDate]
            lsim = LossesSim(sigmaC, fm.ldngs, debtEval[currentBanks], pdd.dfDD[currentBanks].loc[indexDate],dfRR[currentBanks].loc[indexDate],lossType, distrib, dependLoss= dependLoss)
            covarr = CoVaR(lsim.Lsim)
            self.CoVaR.loc[indexDate] = covarr.CoVaR.loc[Params.q]
            self.CoVaR50.loc[indexDate] = covarr.CoVaR.loc[.5]
            self.CoVaR99.loc[indexDate] = covarr.CoVaR.loc[.99]

            self.ECoVaR.loc[indexDate] = covarr.ECoVaR.loc[Params.q]
            self.ECoVaR50.loc[indexDate] = covarr.ECoVaR.loc[.5]
            self.ECoVaR99.loc[indexDate] = covarr.ECoVaR.loc[.99]
            
            self.VaR.loc[indexDate] = covarr.VaR.loc[Params.q]
            self.VaR99.loc[indexDate] = covarr.VaR.loc[.99]            
            self.VaR50.loc[indexDate] = covarr.VaR.loc[.5]
            
            'Expected Shortfall'
            self.ES.loc[indexDate] = covarr.ES.loc[Params.q]
            self.ES99.loc[indexDate] = covarr.ES.loc[.99]            
            self.ES50.loc[indexDate] = covarr.ES.loc[.5]
            
            self.MES.loc[indexDate] = covarr.MES.loc[Params.q]
            self.MES99.loc[indexDate] = covarr.MES.loc[.99]            
            self.MES50.loc[indexDate] = covarr.MES.loc[.5]

            
            self.DeltaCoVaR.loc[indexDate] = covarr.DeltaCoVaR
            self.Weight.loc[indexDate] = debtEval[currentBanks].div(debtEval['Sys'])
            
            pds = DefaultP(lsim.IndctrD)
            self.pN.loc[indexDate] = pds.p['p1ofN']
            self.pNafter1.loc[indexDate] = pds.p['pNafter1']
            self.pNafter2.loc[indexDate] = pds.p['pNafter2']
            self.POA.loc[indexDate] = pds.sysIndex.loc[:,'POA']
            self.VI.loc[indexDate] = pds.sysIndex.loc[:,'VI']            
            self.SII.loc[indexDate] = pds.sysIndex.loc[:,'SII']      

            'Percentage Contributions'
            #self.Weight = dfDebt.loc[lastDate:firstDate+timedelta(weeks=tw)].div(dfDebt.loc[lastDate:firstDate+timedelta(weeks=tw),'Sys'], axis=0)
            
            self.PCtoES.loc[indexDate] = self.MES.loc[indexDate]*self.Weight.loc[indexDate]/self.ES['Sys'].loc[indexDate] 
            self.PCtoES50.loc[indexDate] = self.MES50.loc[indexDate]*self.Weight.loc[indexDate]/self.ES50['Sys'].loc[indexDate] 
            self.PCtoES99.loc[indexDate] = self.MES99.loc[indexDate]*self.Weight.loc[indexDate]/self.ES99['Sys'].loc[indexDate]             
        

class EEI:
    def __init__(self, ldngs, wts, ERR, k, distrib = 'Normal',printt = False):
        '''
        Parameters
        ----------
        ldngs : TYPE
            DESCRIPTION.
        wts : TYPE
            DESCRIPTION.
        dfERR : TYPE
            DESCRIPTION.
        k : np array
            capital ratio.
        distrib : TYPE, optional
            DESCRIPTION. The default is 'Normal'.
        printt : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        self.sigma_ref = .1
        self.r = 0
        self.k_micro = .07
        
        'Get BM Impact'
        EAD_ref= .1 #benchmark relative weight
        LGD_ref = 1.
        X_refM, self.PD_ref = DD(self.k_micro, self.sigma_bm, self.r)
        
        self.SCD_ref = PD_ref*LGD_ref*EAD_ref
        
        'Set k_i_macro to minimize the diff btw SCD_i and SCD_ref'
        
        f_resid = lambda k_i_macro: self.GetSCD(self.k_micro+k_i_macro) - self.SCD_ref
        


        def GetSCDs(k):
            'Get SCDs at bottom'
            DDs = DD(k, sigma, r)
            IndctrD = self.GetDftSims(ldngs,DDs)
            
            'calculate PDs calling DefaultP'
            pds = DefaultP(IndctrD)
            CPD = pds.cpd
            PDs = pd.DataFrame(data =np.diag(pds.jpd.loc[universe,universe]), index = CPD.columns, columns = ['PD'])
            LGD = 1-ERR
            #pd.DataFrame(data =np.ones_like(PDs), index = CPD.columns, columns = ['LGD']) 

            scd = SocialCost(CPD, PDs, wts, LGD)

        
        def GetDftSims(self,ldngs):
            np.random.seed(1)
        
            dWsim = pd.DataFrame(columns = dfDD.index) 
            IndctrD = pd.DataFrame(0, index=np.arange(SetParams().nSims), columns = dfDD.index) 
        
            'Get Factor Simulations'
            nSims = SetParams().nSims
            if distrib == 'Normal':
                M = np.random.normal(loc=0.0, scale=1.0, size=(nF, nSims))  #Factor
                dZ = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))  # Idiosynch 
                dZc = np.random.normal(loc=0.0, scale=1.0, size=(nA, nSims))
            else:
                M = np.random.standard_t(SetParams().df, size = (nF, nSims))
                dZ = np.random.standard_t(SetParams().df, size=(nA, nSims))
                dZc = np.random.standard_t(SetParams().df, size=(nA, nSims))
            
            for jN, Name in enumerate(dfERR.index):
                'Simulate Factor Loadings'                        
                A = ldngs[jN,:] # factor loadings for firm jN
                
                dW = M.T@A + np.sqrt(1- A.T@A)*dZ[jN,:]
                #RR = dfERR[Name]
    
                if printt == True : print('Simulating:', Name)
                IndctrD[Name][dW <= - dfDD.loc[Name]] = 1
                
            return IndctrD
                    
        
        
        def DD(self, k, sigma, r):
            'k, sigma could be arrays with elements standing for each bank'
            DD = -(-np.log(1.+ k) + (r + 1/2 *sigma**2)) / sigma
            X = - DD
            PD = norm.cdf(X)
            return X, PD


            
'''
###################################

dWi = lsim.dWsim['ABN']
DDi = DDEval['ABN']
RR = lsim.RRsim['ABN']

TC = lsim.TC

plt.figure(3)
plt.scatter(dWi,RR)      
plt.xlabel('dW')
plt.ylabel('RR')
plt.title('ABN')


plt.figure(4)
plt.scatter(dWi.where(dWi< -DDi),RR.where(dWi < -DDi), label='No TCs')
plt.scatter(dWi.where(dWi< -DDi),RR.where(dWi < -DDi)*(1-TC), label='With TCs')      
plt.xlabel('dW')
plt.ylabel('RR')
plt.legend()


print(1- np.mean(RR.where(dWi< -DDi)))
'''

'''
'Scatterplot'
name = 'NIBC'
plt.figure()
plt.scatter(lsim.dWsim[name],lsim.RRsim[name])
plt.xlabel(r'$dW_i$')
plt.ylabel(r'$RR_i$')

scatter_matrix(lsim.dWsim, alpha=0.2, figsize=(6, 6), diagonal="kde");
scatter_matrix(lsim.RRsim, alpha=0.2, figsize=(6, 6), diagonal="kde");
'''

#plt.hist(lsim.Lsim['ABN'][lsim.Lsim['ABN']>0])
#lsim.Lsim['ABN'][lsim.Lsim['ABN']>0].count()



'''
'Do Simulation'
#LL = LossesSim(DataSet)
#plt.scatter(LL.LscDbt['ING Bank'],LL.LscDbt['Sys'])
#sample Calculate CoVaRs 
q = .99 
Covarr = CoVaR(LL.L, LL.Debt, 'Loss')

Covarr.CoVaR.T
Covarr.VaR.T
Covarr.DeltaCoVaR.T
'''

'''
'verify that squared sum of the loadings for each variable sums up to less than one'
for j in range(len(FM.ldngs)):
    print(np.matmul(FM.ldngs[j],FM.ldngs[j].T))
    
FM.fctrs.plot(subplots=True)
'''
 
'''
# try parametryzing the beta distribution 

def betaParams(a, b, muRR, sigmaRR):
    res = np.zeros(2)
    res[0] = a - muRR*(a+b)
    res[1] = (a*b) - sigmaRR*((a+b)**2*(a+b + 1.)) 
    return res[0]**2 + res[1]**2

muRR, sigmaRR = .8, .2

f_betaParams = lambda x: betaParams(x[0], x[1], muRR, sigmaRR)

bbounds = Bounds([0, np.inf], [0., np.inf])

minimize(f_betaParams,[1,1], method='SLSQP', bounds=bbounds)

'''