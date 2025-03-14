# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:00:24 2022

@author: danie
"""

from scipy import optimize
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from myplotstyle import * 

path = r'C:\MyGit\systemicRiskBuffers\images'
#%%

class MertonPD:
    def __init__(self, PD):
        'Loading up the data'
        
        self.r= 0.
        self.pd_base = PD
        self.k_grid= np.linspace(0.01,.5,25)
        
        sigma_sol = np.zeros_like(self.k_grid)
        
        for jK, k in enumerate(self.k_grid):
            f = lambda sigma: self.pd_sigma(self.pd_base, k, sigma)
            sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
            sigma_sol[jK] = sol.root
            
        self.sigma_sol = sigma_sol

    def DD(self, k, sigma):
        r = self.r
        w = .3 'RWA intensity'
        'k, sigma could be arrays with elements standing for each bank'
        DD = (-np.log(1. - w*k) + (r - 1/2 *sigma**2)) / sigma
        PD = norm.cdf(- DD)
        return PD
    
    
    def pd_sigma(self, pd, k, sigma):
        
        PD_merton = self.DD(k,sigma)
        res = pd - PD_merton
        
        return res
 

myClass05 = MertonPD(.05)
myClass02 = MertonPD(.02)
myClass01 = MertonPD(.01)

myplotNLines(myClass01.k_grid, [myClass05.sigma_sol, myClass02.sigma_sol,myClass01.sigma_sol],r'$\hat{\sigma}_i$',[r'$PD=5\%$', r'$PD=2\%$',r'$PD=1\%$'],r'$k_i$','%.2f') 
saveFig(path,'sigma_hat')

#%%

def getPD(CDS, LGD):
    '''
    r : risk-free rate 
    T : maturity of the CDS
    dt: descrete time period  
    '''
    r, dt, T  = .005, 1, 5 #LGD .8
    
    a = (1/r)*(1-np.exp(-T*dt*r) )
    b =  ((1/r)**2) *(1 - (T*dt*r + 1)*np.exp(-T*r*dt))
    'Probability of default:'
    PD = (a*CDS) / (a*LGD + b*CDS)
    'Distance to Default'
    #DD = - norm.ppf(PD)
    return PD  


CDS_grid = np.linspace(50, 800, 25) 

PD_grid = getPD(CDS_grid/(100*100), .8)


myplot(CDS_grid,100*PD_grid,['CDS (bps)','PD (%)'], .1, yLim=False, lineStyle='-')
path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\imagesLog\\"
saveFig(path,'PD_CDS')