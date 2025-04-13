# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:05:04 2025

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from optimalSystemicCapital import PDmodel
from getECost import getECost
from myplotstyle import * 
#from GetSystemicRiskSims import * # ??
from GetImpliedParams import *

from datetime import timedelta

from scipy.stats import norm
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


#%%

def getBaseParams():
    paramsDict = {}
    
    paramsDict['Names'] = ['Bank 1', 'Bank 2']
    
    paramsDict['Sigma'] =  np.array([.1, .2])
    paramsDict['wts'] = np.array([.5, .5])
    paramsDict['LGD'] = np.array([1., 1.])
    paramsDict['Rho'] = np.array([[.9], [.9]])
    paramsDict['rwa_intensity'] =  np.array([.3, .3])
    paramsDict['O-SII rates'] = np.array([.05, .05])
    paramsDict['k_bar'] = .02
    paramsDict['Lbar'] = 0.
    
    'including pillar 2:'
    paramsDict['k_p2r'] = np.array([0., 0.])
    return paramsDict

#%% run point in time

myPD = PDmodel('min ES', paramsDict, True)
'macropru buffers'
k_i_marcro = myPD.dict['k_macro_str']*100
print('k_i_marcro:', k_i_marcro)

#%%  vary sigma and optimize

'''
paramsDict = {}

paramsDict['Names'] = ['Bank 1', 'Bank 2']
paramsDict['wts'] = np.array([.5, .5])
paramsDict['LGD'] = np.array([1., 1.])
paramsDict['Rho'] = np.array([[.9], [.9]])
paramsDict['rwa_intensity'] =  np.array([.3, .3])
paramsDict['O-SII rates'] = np.array([.05, .05])
paramsDict['k_bar'] = .02
paramsDict['Lbar'] = .0
paramsDict['k_p2r'] = np.array([0., 0.])
'''

# Define grid of sigma_1 and sigma_2
sigma_grid = np.linspace(0.01, 0.5, 5)  # 10 points from 0.01 to 0.1
k_i_macro = np.zeros((5, 5))  # Store results
PDsys = np.zeros((5, 5))  # Store results
ESS = np.zeros((5, 5))  # Store results

# Loop over sigma_1 and sigma_2 grids
for i, sigma_1_val in enumerate(sigma_grid):
    for j, sigma_2_val in enumerate(sigma_grid):
        
        # Update paramsDict with current sigma values
        paramsDict['Sigma'] = np.array([[sigma_1_val, sigma_2_val]])
        
        # Run PDmodel
        myPD = PDmodel('min ES', paramsDict, True)
        
        # Store the result
        k_macro = myPD.dict['k_macro_str'] * 100
        PDsys[i,j]  = myPD.dict["PDsys"]
        ESS[i,j] = myPD.dict["ESopt"]
        k_i_macro[i, j] = k_macro[0]

# Create a 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Meshgrid for plotting
sigma_1_mesh, sigma_2_mesh = np.meshgrid(sigma_grid, sigma_grid)

# Plot surface
ax.plot_surface(sigma_1_mesh, sigma_2_mesh, ESS, cmap='viridis')

# Labels
ax.set_xlabel('Sigma_1')
ax.set_ylabel('Sigma_2')
ax.set_zlabel('k_i_macro')
ax.set_title('3D Plot of k_i_macro vs. Sigma_1 and Sigma_2')

plt.show()

#%%
plt.plot(sigma_grid,k_i_macro[:,0], label = 'k_macro_j =0')
plt.plot(sigma_grid,k_i_macro[:,4], label = 'k_macro_j =0.05')
plt.legend()


#%% vary k1, k2  and evaluate

'''
paramsDict = {}

paramsDict['Sigma'] =  np.array([.01, .05])

paramsDict['Names'] = ['Bank 1', 'Bank 2']
paramsDict['wts'] = np.array([.5, .5])
paramsDict['LGD'] = np.array([.8, .8])
paramsDict['Rho'] = np.array([[.5], [.5]])
paramsDict['rwa_intensity'] =  np.array([.3, .3])
paramsDict['O-SII rates'] = np.array([.05, .05])
paramsDict['k_bar'] = .02
paramsDict['Lbar'] = .5
paramsDict['k_p2r'] = np.array([0., 0.])
'''

paramsDict = getBaseParams()

# Define grid of sigma_1 and sigma_2
n = 10
k_macro  = np.linspace(0., 0.1, n)  # 10 points from 0.01 to 0.1
#k_i_macro = np.zeros((n, n))  # Store results
PDsys = np.zeros((n, n))  # Store results
ESS = np.zeros((n, n))  # Store results
MES_i = np.zeros((n, n))  # Store results
MES_j = np.zeros((n, n))  # Store results
PD_i = np.zeros((n, n))  # Store results
PD_j = np.zeros((n, n))  # Store results

# Loop over sigma_1 and sigma_2 grids
for i, k_macro_1_val in enumerate(k_macro):
    for j, k_macro_2_val in enumerate(k_macro):
        # Update paramsDict with current sigma values
        paramsDict['O-SII rates'] = np.array([k_macro_1_val, k_macro_2_val])
        # Run PDmodel
        myPD = PDmodel('evaluate ES', paramsDict, True)
        # Store the result
        PDsys[i,j]  = myPD.dict["PDsys"]
        ESS[i,j] = myPD.dict["ESopt"]
        MES = myPD.dict["MESopt"]
        MES_i[i,j] = MES[0]
        MES_j[i,j] = MES[1]
        PD =  myPD.dict["PD"]
        PD_i[i,j] = PD[0]
        PD_j[i,j] = PD[1]


# Meshgrid for plotting
k_macro_1_mesh, k_macro_2_mesh = np.meshgrid(k_macro, k_macro)

# Create the contour plot
fig, ax = plt.subplots(figsize=(6, 4))

# Contour lines
contour = ax.contour(k_macro_1_mesh, k_macro_2_mesh, PDsys, colors='black') #levels=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# Add labels to contour lines
ax.clabel(contour, inline=True, fontsize=8)

# Add the straight line: 0.5 * k_macro_1 + 0.5 * k_macro_2
k_macro_line = np.linspace(0, 0.1, 100)  # Fine grid for smooth line
line_values = 0.5 * k_macro_line + 0.5 * k_macro_line  # Equation for the line
ax.plot(k_macro_line, line_values, 'r--', label=r'$0.5 k_{1,macro} + 0.5 k_{2,macro}$')

line_values = 0.1 - k_macro_line  # Compute k_2 values
ax.plot(k_macro_line, line_values, 'r--', label=r'$0.5 k_{1,macro} + 0.5 k_{2,macro}$')


# Labels and title
ax.set_xlabel(r'$k_{1,macro}$')
ax.set_ylabel(r'$k_{2,macro}$')

# Match the aspect ratio and limit ranges
ax.set_aspect('equal')

plt.show()

#%%

plt.plot(k_macro,ESS[:,0], label = 'k_macro_j =0')
plt.plot(k_macro,ESS[:,5], label = 'k_macro_j =0.05')
plt.legend()


#%%
plt.plot(k_macro,PD_i[:,0], label = 'k_macro_j =0')
plt.plot(k_macro,PD_j[0,:], label = 'k_macro_j =0.05')
plt.legend()


#%%
# Meshgrid for plotting
k_macro_1_mesh, k_macro_2_mesh = np.meshgrid(k_macro, k_macro)


# Create a 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(k_macro_1_mesh, k_macro_2_mesh, ESS, cmap='viridis', alpha=0.7)

# Labels
ax.set_xlabel('k_1_macro')
ax.set_ylabel('k_2_macro')
ax.set_zlabel('ESS')
ax.set_title('ESS')

plt.show()

# Create a contour plot
fig, ax = plt.subplots(figsize=(8, 6))

# Filled contour plot
contour = ax.contourf(k_macro_1_mesh, k_macro_2_mesh, ESS, levels=20, cmap='viridis')

# Add contour lines
ax.contour(k_macro_1_mesh, k_macro_2_mesh, ESS, levels=10, colors='black', linewidths=0.5)

# Colorbar
cbar = plt.colorbar(contour)
cbar.set_label('ESS')

# Labels
ax.set_xlabel('k_1_macro')
ax.set_ylabel('k_2_macro')
ax.set_title('ESS Contour Plot')

plt.show()

#%%%

