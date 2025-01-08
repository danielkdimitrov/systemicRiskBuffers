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
        if varyParam == 'EEI opt w/data' or varyParam == 'min ES' or varyParam == 'EEI opt vary SCDref':
            self.rho_base = rho_base['Rho']
            self.nBanks, self.nF = self.rho_base.shape #number of factors             
            self.U_base = self.factorSim(self.rho_base)
            self.Lbar = rho_base['Lbar']
            
            'Reference bank : '
            self.sigma_ref = np.mean(rho_base['Sigma']) #why max?? maybe average?? 
            self.EAD_ref = np.min( rho_base['wts'] ) #.01 #1/self.nBanks #
            self.LGD_ref = np.mean(rho_base['LGD']) 
            if useP2R == True:
                self.k_micro =  np.array([k_micro]*self.nBanks) + rho_base['k_p2r']
            else:
                self.k_micro =  np.array([k_micro]*self.nBanks) 
        else:
            self.nF = 1 #number of factors 
            self.rho_base = rho_base            
            self.nBanks = 10 #number of banks
            self.Lbar = 0
            self.k_micro =  np.array([k_micro]*self.nBanks)                        
            self.nGrdPnts = 25
            self.Sigma_base = np.array([.1]*self.nBanks)
                                    
            'Reference bank : '
            self.sigma_ref = .1
            self.EAD_ref = .1
            self.LGD_ref = .2
                   
        self.LGD =  np.array([1.]*self.nBanks)
        self.EAD = np.array([1/self.nBanks]*self.nBanks)
     
        self.U_base = self.factorSim(self.rho_base)
        self.r = 0.

        'using the first institution as a reference'
        X_refM, self.PD_ref = self.DD(self.k_micro[0], self.sigma_ref, self.r)
        
        self.SCD_ref = self.PD_ref*self.LGD_ref*self.EAD_ref        #.025
        
        '--------- Vary Rho'
        if varyParam == 'rho': 
            rho_grid = np.linspace(0,.99, self.nGrdPnts)
            self.dict = self.getVaryRho(rho_grid)
        '--------- Vary k ---------'            
        if varyParam == 'k':
            k_macro_grid = np.linspace(0,.25,self.nGrdPnts)
            self.dict = self.getVarryK(k_macro_grid)
        '--------- Vary w ---------'
        if varyParam == 'w':
            w_grid = np.linspace(0.1,.9, self.nGrdPnts)
            self.dict = self.getVaryW(w_grid)
        '---- Evaluate EEI theory -----'
        if varyParam == 'EEI':
            k_macro_grid = np.linspace(0,.25,self.nGrdPnts)
            #k_micro_i = .07            
            self.dict = self.getVaryEEI(k_macro_grid)
            'Find the solution'
            self.dict['k_i_str'] = self.getEEIstarOneBnak(self.k_micro,self.U_base, self.EAD) 
        if varyParam == 'EEI Multivariate':
            self.EAD = np.concatenate((np.array([.5, .2]), np.ones(8)*(.3/8)), axis=0)
            #self.dict['k_str'] = self.getEEImulti(self.k_micro,self.U_base, self.EAD) 
            k1_macro_grid = np.linspace(0,.25,self.nGrdPnts)
            k2_macro_grid = np.linspace(0,.25,self.nGrdPnts)
            self.dict = self.getVaryEEImulti(k1_macro_grid,k2_macro_grid)
        if varyParam == 'EEI opt Multivariate':
            self.dict = {}
            self.EAD = np.concatenate((np.array([.5, .2]), np.ones(8)*(.3/8)), axis=0)
            self.sysIndex = np.concatenate((np.array([1, 1]), np.zeros(8)), axis=0) ==1  # index of which institutions are systemic
            self.nOpt = np.count_nonzero(self.sysIndex) #number of systemic institutions
            'rootfining here :'
            x0 = np.array([.105,.101]) #np.zeros(self.nOpt)
            x1 = np.array([.0,.0]) #np.zeros(self.nOpt)
            
            optns = {'ftol ':10**(-9) }
            f = lambda k_macro: np.sum(self.getResidMulti(k_macro )**2)
            
            sol = minimize(f, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
            
            self.sol = sol
            print('sol:', sol)
            self.dict["k_macro_str"] = sol.x
            
            self.dict["k_str"] = self.k_micro[self.sysIndex] + self.dict["k_macro_str"]     
            self.dict["IndirectCost"], self.dict["DirectCost"], self.dict["SCD"] = self.getResidMulti(sol.x,True)
            self.dict["Names"] = rho_base['Names']
            #self.dict['k_str'] = self.getEEImulti(self.k_micro,self.U_base, self.EAD) 
        if varyParam == 'EEI opt w/data':
            'in this case rho base is a dictionary with all the necessary params'
            self.EAD = rho_base['wts']
            self.Sigma_base = rho_base['Sigma']
            self.Names = rho_base['Names']
            self.nOpt = len(rho_base['Names'])
            self.sysIndex = np.ones(self.nOpt)  ==1  # index of which institutions are systemic
            
            'Add fn that does this and call it '
            self.dict = self.getKmacroEEI()

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
            x0 = np.ones(self.nOpt)*self.k_bar/self.nOpt
            f = lambda k_macro: self.getES(k_macro, self.U_base)
            f_constr = lambda k_macro: np.sum(k_macro*self.EAD) - self.k_bar 
            dict_constr = {'type':'eq', 'fun': f_constr}
            sol = minimize(f, x0, method='SLSQP',constraints=dict_constr)# , options={'xatol': 1e-15, 'disp': False})
            self.sol = sol
            print('sol:', sol)
            print('constr:', f_constr(sol.x))
            self.dict["k_macro_str"] = sol.x           
            self.dict["k_str"] = self.k_micro[self.sysIndex] + self.dict["k_macro_str"]     
            self.dict["ESopt"], self.dict['MESopt'] = self.getES(self.dict["k_macro_str"],self.U_base,True)
            self.dict["ESmicro"], self.dict['MESmicro'] = self.getES(np.zeros(self.nOpt) ,self.U_base,True)
            self.dict["IndirectCost"], self.dict["DirectCost"], self.dict["SCD"] = self.getResidMulti(sol.x,True)
            
            #self.dict["EScurr"], self.dict['MEScurr'] = self.getES(self.current_k,self.U_base,True) #k_current here completely depends on O-SII. Need to change that to reuse
          
            'get PD sys at final solution:'
            Xx, self.dict["PD"] = self.DD(self.dict["k_str"], self.Sigma_base, self.r)
            IndD, IndS = self.defaultSims(self.U_base,Xx)
            L = IndD.T*self.LGD
            Lsys = np.sum(L*self.EAD.T,axis=1)
            self.dict["PDsys"] = len(Lsys[Lsys>self.Lbar]) / len(Lsys)
            self.Lsys = Lsys
            
        if varyParam == 'EEI opt vary SCDref':
            'vary the SCD'
            'in this case rho base is a dictionary with all the necessary params'
            self.EAD = rho_base['wts']
            self.Sigma_base = rho_base['Sigma']
            self.Names = rho_base['Names']
            self.nOpt = len(rho_base['Names'])
            self.sysIndex = np.ones(self.nOpt)  ==1  # index of which institutions are systemic
            self.dict = {}
            
            'construct a refeernce'
            X_refM, self.PD_ref = self.DD(self.k_micro[0], self.sigma_ref, self.r)        
            EAD_grid = np.linspace(0.01,.1,10)     
            'loop over the grid of EAD '
            for jS, EADi in enumerate(EAD_grid):
                'EADi is w_ref'
                print('EAD_ref:', EADi)
                self.dict[EADi] = {}
                self.SCD_ref = self.PD_ref*self.LGD_ref*EADi #SCD_ref

                self.dict[EADi] = self.getKmacroEEI()
        'Socially Optimal function'
        if varyParam == 'Social Opt':
            'get the socially optimal '
            self.Lambda = .18
            self.Eta = .024
            self.K_bar = np.linspace(0,.2,20) #  0.05,.1            
            self.ECost, self.dfPDsys, self.dfES, self.SCB, self.k_bar_min = getECost()
        
                
    def getKmacroEEI(self):
        
        print('EAD : ', self.EAD)
        myDict = {}
            
        'construct a refeernce'
        k_micro_ref = np.median(self.k_micro)
        X_refM, myDict['PD_ref'] = self.DD(k_micro_ref, self.sigma_ref, self.r)        
        myDict['SCD_ref'] = myDict['PD_ref']*self.LGD_ref*self.EAD_ref        

        'construct the starting point'
        x0 = np.zeros(self.nOpt)
        for j in range(self.nOpt):
            x0[j] = self.getEEIstarOneBnak(self.k_micro,self.U_base, self.EAD, j)
        myDict['k_macro_str_univ'] = x0
        myDict["IndirectCost_univ"], myDict["DirectCost_univ"], myDict["SCD_univ"] = self.getResidMulti(x0,True)
        print('Starting point, k_macro_str_univ:', x0)
        print('Starting point, SCD', myDict["SCD_univ"])   #add SCD w/o macroprud buffers              
            
        'run joint optimization'
        #x0 = np.ones(self.nOpt)*.05 #np.zeros(self.nOpt)
        f = lambda k_macro: np.sum(self.getResidMulti(k_macro )**2)
        'minimize SSqErs'
        sol = minimize(f, x0, method='nelder-mead',options={'xatol': 1e-15, 'disp': True})
        #print(self.getResidMulti(sol.x,True))            
        #minimizer_kwargs = {"method": "BFGS"}
        #ret = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,niter=200)
        #print("global minimum: x = %.4f, f(x) = %.4f" % (ret.x, ret.fun))
        #self.sol = sol
        print('sol:', sol)
        myDict["k_macro_str"] = sol.x           
        myDict["k_str"] = self.k_micro[self.sysIndex] + myDict["k_macro_str"]     
        myDict["IndirectCost"], myDict["DirectCost"], myDict["SCD"] = self.getResidMulti(myDict["k_macro_str"],True)
        Xx, myDict["PD"]  = self.DD( myDict["k_str"], self.Sigma_base, self.r)
        
        print('SCD at optimum :', myDict["SCD"])
        print('k_macro at optimum :', myDict["k_macro_str"], '\n')
        
        
        return myDict
                
    def getVaryRho(self,rho_grid):
        PDdict = {}
        PDdict["PDi"], PDdict["PDj_i"], PDdict["IndirectCost"], PDdict["DirectCost"], PDdict['SCDi'] = np.zeros_like(rho_grid), np.zeros_like(rho_grid), np.zeros_like(rho_grid), np.zeros_like(rho_grid), np.zeros_like(rho_grid)
        PDdict["ENd"], PDdict["ENd_i"], PDdict["rho"] = np.zeros_like(rho_grid), np.zeros_like(rho_grid), np.zeros_like(rho_grid)
        PDdict["k_i_str"] = np.zeros_like(rho_grid)
        
        for jRho, rho in enumerate(rho_grid):
            U = self.factorSim(rho)
            Xrho, PDrho = self.DD(self.k_micro, self.Sigma_base, self.r)
            IndD, IndS = self.defaultSims(U,Xrho)
            PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS)
            IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD)
            
            
            PDdict['PDi'][jRho], PDdict['PDj_i'][jRho], PDdict['IndirectCost'][jRho], PDdict["DirectCost"][jRho], PDdict['SCDi'][jRho]  = PD[0], PDj_i[1], IndirectCost, DirectCost, SCDi
            PDdict["ENd"][jRho], PDdict["ENd_i"][jRho] = ENd, ENd_i
            PDdict["rho"][jRho] = rho
            PDdict["k_i_str"][jRho] = self.getEEIstarOneBnak(self.k_micro,U, self.EAD) 

        return PDdict
    
    def getVarryK(self,k_macro_grid):
            
        'Simulate Factors'
        #U = self.U_base
                
        PDdict = {}
        PDdict["PDi"], PDdict["PDj_i"], PDdict["IndirectCost"], PDdict["DirectCost"], PDdict['SCDi'] = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["ENd"], PDdict["ENd_i"], PDdict["k_i"]  = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["k_i_str"] = np.zeros_like(k_macro_grid)

        k_i = self.k_micro.copy() 

        for jK, k_macro in enumerate(k_macro_grid):
            'vary the capitalization of the first bank'
            k_i[0] = self.k_micro[0] + k_macro 
                        
            Xk, PDk = self.DD(k_i, self.Sigma_base, self.r)
            IndD, IndS = self.defaultSims(self.U_base,Xk)
            PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS)
            IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD)
            
            
            PDdict['PDi'][jK], PDdict['PDj_i'][jK], PDdict['IndirectCost'][jK], PDdict["DirectCost"][jK], PDdict['SCDi'][jK]  = PD[0], PDj_i[1], IndirectCost, DirectCost, SCDi
            PDdict["ENd"][jK], PDdict["ENd_i"][jK] = ENd, ENd_i
            PDdict['k_i'][jK] = k_i[0]
            'increasing the micropdurential buffers for all'
            PDdict["k_i_str"][jK] = self.getEEIstarOneBnak(self.k_micro + k_macro, self.U_base, self.EAD) 

        return PDdict
             
    def getVaryW(self,w_grid):
    
        PDdict = {}
        PDdict["PDi"], PDdict["PDj_i"], PDdict["IndirectCost"], PDdict["DirectCost"], PDdict['SCDi'] = np.zeros_like(w_grid), np.zeros_like(w_grid), np.zeros_like(w_grid), np.zeros_like(w_grid), np.zeros_like(w_grid)
        PDdict["ENd"], PDdict["ENd_i"] = np.zeros_like(w_grid), np.zeros_like(w_grid)
        PDdict["w_i"] = np.zeros_like(w_grid)
        PDdict["k_i_str"] = np.zeros_like(w_grid)        
        'Vary the weights'
        EAD_now = self.EAD.copy()
        'Simulate Factors'
        Xrho, PDrho = self.DD(self.k_micro, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(self.U_base,Xrho)
        PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS)
        
        for jRho, w_i in enumerate(w_grid):
            EAD_now[0] = w_i
            EAD_now[1:] = self.EAD[1:]*(1-w_i)/(1-self.EAD[0])
            IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, EAD_now)
            
            
            PDdict['PDi'][jRho], PDdict['PDj_i'][jRho], PDdict['IndirectCost'][jRho], PDdict["DirectCost"][jRho], PDdict['SCDi'][jRho]  = PD[0], PDj_i[1], IndirectCost, DirectCost, SCDi
            PDdict["ENd"][jRho], PDdict["ENd_i"][jRho] = ENd, ENd_i
            PDdict["w_i"][jRho] = w_i
            PDdict["k_i_str"][jRho] = self.getEEIstarOneBnak(self.k_micro, self.U_base, EAD_now)             
            
            'evaluate k^*'
            
        return PDdict
    
    def getVaryEEI(self, k_macro_grid):
        'initialize'
        PDdict = {}
        PDdict["PDi"], PDdict["PDj_i"], PDdict["IndirectCost"], PDdict["DirectCost"], PDdict['SCDi'] = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["ENd"], PDdict["ENd_i"] = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["k_i"] = np.zeros_like(k_macro_grid)
        
        'evaluate the BM at the microprudential buffers'
        X_base, PD_base = self.DD(self.k_micro, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(self.U_base,X_base)
        PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS)
        IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD)        
        PDdict["SCDiBM"] = DirectCost

        'evaluate i with the macroprudential add-on'        
        k_i = self.k_micro.copy()
        for jK, k_macro in enumerate(k_macro_grid):
            'vary the capitalization of the first bank'
            k_i[0] = self.k_micro[0] + k_macro 
                        
            Xk_macro, PDk_macro = self.DD(k_i, self.Sigma_base, self.r)
            IndD_macro, IndS_macro = self.defaultSims(self.U_base,Xk_macro)
            PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD_macro, IndS_macro)
            IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD)
            
            
            PDdict['PDi'][jK], PDdict['PDj_i'][jK], PDdict['IndirectCost'][jK], PDdict["DirectCost"][jK], PDdict['SCDi'][jK]  = PD[0], PDj_i[1], IndirectCost, DirectCost, SCDi
            PDdict["ENd"][jK], PDdict["ENd_i"][jK] = ENd, ENd_i
            PDdict['k_i'][jK] = k_i[0]
        
        return PDdict        

    def getVaryEEImulti(self, k1_macro_grid, k2_macro_grid):
        'initialize'
        PDdict = {}
        PDdict["PD1"], PDdict["PD2_1"], PDdict["IndirectCost1"], PDdict["DirectCost1"], PDdict['SCD1'] = np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts])
        #PDdict["ENd"], PDdict["ENd_i"] = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["k1"] = np.zeros_like(k1_macro_grid)
        
        PDdict["PD2"], PDdict["PD1_2"], PDdict["IndirectCost2"], PDdict["DirectCost2"], PDdict['SCD2'] = np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts]), np.zeros([self.nGrdPnts,self.nGrdPnts])
        #PDdict["ENd"], PDdict["ENd_i"] = np.zeros_like(k_macro_grid), np.zeros_like(k_macro_grid)
        PDdict["k2"] = np.zeros_like(k2_macro_grid)
        
        
        'evaluate the BM at the microprudential buffers'
        X_base, PD_base = self.DD(self.k_micro, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(self.U_base,X_base)
        PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS)
        IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD)        
        PDdict["SCDiBM"] = DirectCost

        'evaluate i with the macroprudential add-on'        
        k = self.k_micro.copy()
        
        for jK1, k1_macro in enumerate(k1_macro_grid):
            PDdict['k1'][jK1] = k1_macro
            for jK2, k2_macro in enumerate(k2_macro_grid):
            
                'vary the capitalization of the first and bank'
                k[0] = self.k_micro[0] + k1_macro
                k[1] = self.k_micro[1] + k2_macro 
                
                'evaluate PDs at the new capital requirements'            
                Xk_macro, PDk_macro = self.DD(k, self.Sigma_base, self.r)
                IndD_macro, IndS_macro = self.defaultSims(self.U_base,Xk_macro)
                
                'bank 1'
                PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD_macro, IndS_macro, i = 0)
                IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD, i = 0)                
                PDdict['PD1'][jK1,jK2], PDdict['PD2_1'][jK1,jK2], PDdict['IndirectCost1'][jK1,jK2], PDdict["DirectCost1"][jK1,jK2], PDdict['SCD1'][jK1,jK2]  = PD[0], PDj_i[1], IndirectCost, DirectCost, SCDi
                #PDdict["ENd"][jK1,jK2], PDdict["ENd_i"][jK1,jK2] = ENd, ENd_i
                
                'bank 2'
                PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD_macro, IndS_macro, i = 1)
                IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, self.EAD, i = 1)                
                PDdict['PD2'][jK1,jK2], PDdict['PD1_2'][jK1,jK2], PDdict['IndirectCost2'][jK1,jK2], PDdict["DirectCost2"][jK1,jK2], PDdict['SCD2'][jK1,jK2]  = PD[1], PDj_i[0], IndirectCost, DirectCost, SCDi
                #PDdict["ENd"][jK2], PDdict["ENd_i"][jK2] = ENd, ENd_i
                PDdict['k2'][jK2] = k2_macro                
        
        return PDdict        
                
        
    def getResidMulti(self,k_macro, getSCDs=False):
        'k_macro now is an array 2x1'
        'use sysIndex : index of the variables which are deemed systemic'
        'set macro capital requirements for the systemic banks'
        k_total = self.k_micro.copy()
        k_total[self.sysIndex] = self.k_micro[self.sysIndex] + k_macro
        #print('k_macro',k_macro)
        Xx, PDrho = self.DD(k_total, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(self.U_base,Xx)
        resid = np.zeros(self.nOpt)
        IndirectCost, DirectCost, SCDi = np.zeros(self.nOpt), np.zeros(self.nOpt), np.zeros(self.nOpt)
        IndexSystemic = np.where(self.sysIndex==True)         
        for i in range(self.nOpt):
            jSys = IndexSystemic[0][i]
            PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS, jSys)        
            IndirectCost[i], DirectCost[i], SCDi[i] = self.GetSCD(PD, PDj_i, self.LGD, self.EAD, jSys) 
            resid[i] = SCDi[i] - self.SCD_ref
        if getSCDs == True:
            return IndirectCost, DirectCost, SCDi
        #print('SCD :', SCDi)
        sumSqrtResid = np.sum(resid**2)
        #print('residual :', resid)
        '''
        print('PD',PD)
        print('resid', resid,'sumSqResid', sumSqrtResid)
        print('SCDi ',SCDi, '\n')
        '''
        return resid


    
    def getEEIstarOneBnak(self, k_micro , U ,  EAD, i=0 ):        
        
        'Set k_i_macro for one bank to minimize the diff btw SCD_i and SCD_ref'
        f = lambda k_i_macro: self.Resid(k_i_macro, k_micro, U, EAD, i)
        k_i_macro = bisect(f,-.05,.5) #root_scalar(self.Resid, bracket=[0.01, 1], method='bisect') #k_micro[0]
        return k_i_macro              
        
    def Resid(self,k_i_macro, k_micro, U, EAD, i=0):
        'i: the sysemmic player'
        k_i = k_micro.copy()
        'set macro capital requirements for the first bank'
        k_i[i] = k_i[i]+k_i_macro      
        Xx, PDrho = self.DD(k_i, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(U,Xx)
        PD, PDj_i, ENd, ENd_i = self.defaultCrossCorrs(IndD, IndS, i)
        
        IndirectCost, DirectCost, SCDi = self.GetSCD(PD, PDj_i, self.LGD, EAD, i)
        #print('k_i_macro', k_i_macro)
        #print("SCD Ref:", self.SCD_ref, "SCDi", SCDi)
        resid = SCDi - self.SCD_ref
        return resid 
            
    def DD(self, k, sigma, r):
        'k, sigma could be arrays with elements standing for each bank'
        DD = (-np.log(1. - k) + (r - 1/2 *sigma**2)) / sigma
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
    
    def getES(self, k_macro, U, showMES = False):

        k_total = self.k_micro.copy()
        k_total[self.sysIndex] = self.k_micro[self.sysIndex] + k_macro
        #print('k_macro',k_macro)
        Xx, PDrho = self.DD(k_total, self.Sigma_base, self.r)
        IndD, IndS = self.defaultSims(U,Xx)
        
        MES = self.getMES(IndD)
        ESsys = np.sum(self.EAD* MES)
        if showMES == True:
            return ESsys, MES
        
        return ESsys

    def getMES(self, IndD):
        L = IndD.T*self.LGD
        Lsys = np.sum(L*self.EAD.T,axis=1)
        MES = np.average(L[Lsys>self.Lbar],axis=0)
        
        return MES



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
