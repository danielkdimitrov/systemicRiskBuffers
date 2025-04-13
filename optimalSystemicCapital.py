# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:55:02 2022

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from datetime import timedelta

from scipy.stats import norm
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest
from scipy.optimize import root_scalar, bisect, minimize, basinhopping

from myplotstyle import * 


class PDmodel:
    def __init__(self, varyParam, rho_base = .9, paramsDict = False, useP2R = True, k_micro =0.07):       
        '''
        Get Implied Probab of Default, and Distance to Default from CDS prices
        INPUT : 
            varyParam : string 
            - rho
            - 
            - 
        OUTPUT : 
            dfPD : dataframe - Implied (from CDS rates) Probability of Default
            dfDD : dataframe - Implied Distance to Default 
        '''
        self.nSims = 10**5 #number of simulations
        
        'base case'
        self.rho_base = rho_base['Rho']
        self.nBanks, self.nF = self.rho_base.shape #number of factors             
        self.U_base = self.factorSim(self.rho_base)
        self.Lbar = rho_base['Lbar']        
        self.k_micro =  np.array([k_micro]*self.nBanks) + rho_base['k_p2r']
        self.LGD = rho_base['LGD']  #np.array([1.]*self.nBanks)
        # = np.array([1/self.nBanks]*self.nBanks)
        self.r = 0.
        self.rwa_intensity = rho_base['rwa_intensity']
        '---- Min ES -----'
        if varyParam == 'min ES':
            self.EAD = rho_base['wts']
            self.Sigma_base = rho_base['Sigma']
            self.Names = rho_base['Names']
            self.nOpt = len(rho_base['Names'])
            self.sysIndex = np.ones(self.nOpt)  ==1  # index of which institutions are systemic
            #self.current_k = rho_base['O-SII rates']
            self.k_bar = rho_base['k_bar']
            self.dict = {}
            
            'sum squared residuals'
            '''
            x0 = [0., 0.1]
            
            #print(x0)
            dict_constr = {'type':'eq', 'fun': f_constr}
            sol = minimize(f, x0, method='SLSQP',constraints=dict_constr,options={'finite_diff_rel_step': 1e-6})# , options={'xatol': 1e-15, 'disp': False})
            self.sol = sol
            
            # Inequality constraint: sum(k_macro * EAD) - k_bar >= 0
            ineq_constraint = {
                'type': 'ineq',
                'fun': lambda k_macro: np.sum(k_macro * self.EAD) - self.k_bar
            }
            ''' 
            # Objective
            f = lambda k_macro: self.getES(k_macro, self.U_base)
            
            # Inequality constraint: sum(k_macro * EAD) <= k_bar
            f_constr = lambda k_macro: self.k_bar - np.sum(k_macro * self.EAD)
            ineq_constraint = {'type': 'ineq', 'fun': f_constr}
            
            # Non-negativity bounds
            bounds = [(0, None)] * len(self.EAD)
            
            # Initial guess
            x0 = [0.0] * len(self.EAD)
            
            # Run optimization
            sol = minimize(f, x0, method='SLSQP', bounds=bounds,
                           constraints=[ineq_constraint],
                           options={'disp': True, 'maxiter': 1000})
            print('sol:', sol)
            print('constr:', f_constr(sol.x))
            self.dict["k_macro_str"] = sol.x           
            self.dict["k_str"] = self.k_micro[self.sysIndex] + self.dict["k_macro_str"]     
            self.dict["ESopt"], self.dict['MESopt'] = self.getES(self.dict["k_macro_str"],self.U_base,True)
            self.dict["ESmicro"], self.dict['MESmicro'] = self.getES(np.zeros(self.nOpt) ,self.U_base,True)
                      
            'get PD sys at final solution:'
            Xx, self.dict["PD"] = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
            IndD, IndS = self.defaultSims(self.U_base,Xx)
            L = IndD.T*self.LGD
            Lsys = np.sum(L*self.EAD.T,axis=1)
            self.dict["PDsys"] = len(Lsys[Lsys>self.Lbar]) / len(Lsys)
            self.Lsys = Lsys
        'Socially Optimal function'
        if varyParam == 'Social Opt':
            'get the socially optimal '
            self.Lambda = .18
            self.Eta = .024
            self.K_bar = np.linspace(0,.2,20) #  0.05,.1            
            self.ECost, self.dfPDsys, self.dfES, self.SCB, self.k_bar_min = getECost()
        
        if varyParam == 'evaluate ES':
            self.EAD = rho_base['wts']
            self.Sigma_base = rho_base['Sigma']
            self.Names = rho_base['Names']
            self.nOpt = len(rho_base['Names'])
            self.sysIndex = np.ones(self.nOpt)  ==1  # index of which institutions are systemic
            self.k_bar = rho_base['k_bar']
            k_macro_input = rho_base['O-SII rates']
            self.dict = {}
                        
            self.dict["k_macro_str"] = k_macro_input           
            self.dict["k_str"] = self.k_micro[self.sysIndex] + self.dict["k_macro_str"]     
            self.dict["ESopt"], self.dict['MESopt'] = self.getES(k_macro_input, self.U_base,True)
            self.dict["ESmicro"], self.dict['MESmicro'] = self.getES(np.zeros(self.nOpt) ,self.U_base,True)            
            #self.dict["EScurr"], self.dict['MEScurr'] = self.getES(self.current_k,self.U_base,True) #k_current here completely depends on O-SII. Need to change that to reuse
          
            'get PD sys at final solution:'
            Xx, self.dict["PD"] = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
            IndD, IndS = self.defaultSims(self.U_base,Xx)
            L = IndD.T*self.LGD
            Lsys = np.sum(L*self.EAD.T,axis=1)
            self.dict["PDsys"] = self.getPDsys(Lsys) 
            self.Lsys = Lsys
            
    def DD(self,uspilon, k, sigma, r):
        'k, sigma could be arrays with elements standing for each bank'
        #print(sigma)
        DD = (-np.log(1. - uspilon*k) + (r - 1/2 *sigma**2)) / sigma
        X = - DD #default threshold
        PD = norm.cdf(X)
        return X, PD
    
    
    def factorSim(self, rho):
        'Simulate Facotrs'
        np.random.seed(1)                   
        sims = np.random.normal(loc=0.0, scale=1.0, size=(self.nF + self.nBanks, self.nSims)) #this needs to work for more than one factors
        M =  sims[:self.nF,:] #Factor
        dZ =  sims[self.nF:,:]# Idiosynch
        
        if self.nF == 1:
            U = M*rho + np.sqrt(1- rho**2)*dZ        
        else:
            U = np.zeros([self.nBanks, self.nSims])
            'loop through all banks'
            for jRho in range(self.nBanks):
                rho_j = rho[jRho,:]
                U[jRho,:]= M.T@rho_j + np.sqrt(1- rho_j.T@rho_j)*dZ[jRho,:] #here rho is a matrix array
            
        return U
    
    def defaultSims(self,U,Xx):
        '''        
        Parameters
        ----------
        U : Array
            Simulated Latent variable.
        Xx : Array
            Default Threshold.

        Returns 
        -------
        None.

        '''
        'Initialize'
        IndD = np.zeros([self.nBanks, self.nSims]) # default indicator per bank
        IndS = np.zeros(self.nSims) # systemic crisis indicator
        
        IndD[U <= Xx.reshape(self.nBanks,1)] = 1 #default indicator
        
        'Calculate systemic metrics'
        #IndS[Nd > 1] = 1 
            
        'Calculate default probab conditional on sys crisis'
        #i = 0
        #IndDi_s = IndD[i,IndS ==1] # indicator default of i conditional on s

        #PD_s = np.average(IndS) #
        #ENd = np.average(Nd[IndS==1])
        
        return IndD, IndS

        
    def defaultCrossCorrs(self, IndD, IndS, i = 0, getPDsys = False):
        ''''
        Calculate default simulations
        IndD : indicator of default 
        IndS : indicator of systemic distress        
                this is legacy, but remove later
        i : indicator of the bank to be evaluated
        '''
        
        PD = np.average(IndD,1) # vector (array) of default probabs
                
        PDj_i = np.zeros(self.nBanks) #all banks but the reference one
        Nd = np.sum(IndD,0) # number of defaults per scerio
        L = IndD.T*self.LGD
        Lsys = np.sum(L*self.EAD.T,axis=1)
        #ELi = 
        #ELsys =
        ELi_sys = np.average(L[Lsys>self.Lbar],axis=0)

        'conditional on i-th bank default. later could be in a loop:'
        
        ENd = np.average(Nd)
        ENd_i = np.average( Nd[IndD[i,:]  == 1]  ) #
        for j in range(0,self.nBanks):
            'j-s default conditional on i'
            PDj_i[j] = np.average(IndD[j,(IndD[i] ==1)]) # (IndS ==1) &

            
        ''' 
        Ei_s : Direct Cost
        sum_j Ej_si : Cost of impact
        '''  
        #sumEj_si = np.sum(Ej_si)/self.nBanks
        return PD, PDj_i, ENd, ENd_i
    
    def GetSCD(self,PD, PDj_i, LGD, EAD, i=0):
        PD_j = np.delete(PD,i)
        EAD_j = np.delete(EAD,i)
        LGD_j  = np.delete(LGD,i)
        PDj_i  = np.delete(PDj_i,i) #drop the reference bank
        if PD[i] == 0:
            IndirectCost = 0.
            DirectCost = 0.
        else:
            IndirectCost =np.sum(EAD_j*LGD_j*PDj_i - EAD_j*LGD_j*PD_j)*PD[i]
            DirectCost = EAD[i]*LGD[i]*PD[i]
        SCDi = IndirectCost + DirectCost
                
        return IndirectCost, DirectCost, SCDi
    
    def getPDsys(self,Lsys):
        PDsys = len(Lsys[Lsys>self.Lbar]) / len(Lsys)
        return PDsys
        
    def getES(self, k_macro, U, showMES = False):
        #print(k_macro)
        k_total = self.k_micro.copy()
        k_total[self.sysIndex] = self.k_micro[self.sysIndex] + k_macro
        #print('k_macro',k_macro)
        Xx, PDrho = self.DD(self.rwa_intensity, k_total, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(U,Xx)
        
        MES, PDsys = self.getMES(IndD)
        ESsys = np.sum(self.EAD* MES)*PDsys
        if showMES == True:
            return ESsys, MES
        
        return PDsys #ESsys*PDsys

    def getMES(self, IndD):
        L = IndD.T*self.LGD
        Lsys = np.sum(L*self.EAD.T,axis=1)
        MES = np.average(L[Lsys>self.Lbar],axis=0)
        PDsys = self.getPDsys(Lsys)
        return MES, PDsys

    def getECost(self):
        'I need to finish this later. I need it to reference itself'
        
        dfES = pd.DataFrame(index = K_bar, columns = ['ES'])
        dfMES = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
        dfKimacro = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
        dfPD = pd.DataFrame(index = K_bar, columns = paramsDict['Names'])
        ECost = pd.DataFrame(index = K_bar, columns = ['Sys'])
        dfPDsys = pd.DataFrame(index = K_bar, columns = ['Sys'])
        
        
        for jK, k_bar in enumerate(self.K_bar):
            paramsDict['k_bar'] = k_bar
            myPD = PDmodel('min ES', paramsDict, True, True)
            dfES.loc[k_bar] = myPD.dict['ESopt']
            #dfMES.loc[k_bar] = myPD.dict['MESopt']
            dfKimacro.loc[k_bar] = myPD.dict['k_macro_str']
            #dfPD.loc[k_bar] = myPD.dict['PD']
            dfPDsys.loc[k_bar] = myPD.dict['PDsys']
            
                
        'Expected Cost Function'
            
        dfFirstTerm = dfPDsys*self.Lambda*dfES.values
        SCB = self.Eta*(pd.DataFrame(index = K_bar, data = K_bar, columns= ['Sys']).values - 0.)#.07
        dfSecond = (1-dfPDsys)*SCB
        
        ECost = dfFirstTerm  + dfSecond
        
        nMin = np.where(ECost ==  ECost.min())[0]
        k_bar_min = ECost.iloc[nMin]
        return ECost, dfPDsys, dfES, SCB, k_bar_min
