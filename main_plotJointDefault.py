# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:00:16 2025

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from myplotstyle import saveFig

path = r"C:\MyGit\systemicRiskBuffers\images//"

#%%
# Parameters
upsilon1 = 0.3
upsilon2 = 0.3
sigma1 = 0.03
sigma2 = 0.03
r = 0.0
rho = 0.6
k_1 = .1
k_2 = .1

#%% 

def PD(k, upsilon, sigma, r):
    """Default probability for given capital ratio k"""
    argument = (np.log(1 - upsilon * k) - (r - 0.5 * sigma**2)) / sigma
    return norm.cdf(argument)

def joint_default_prob(k1, k2, upsilon1, upsilon2, sigma1, sigma2, r, rho):
    """Probability of joint default using Vasicek correlation structure"""
    PD1 = PD(k1, upsilon1, sigma1, r)
    PD2 = PD(k2, upsilon2, sigma2, r)
    inv_PD1 = norm.ppf(PD1)
    inv_PD2 = norm.ppf(PD2)
    
    # Bivariate normal CDF
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return multivariate_normal.cdf([inv_PD1, inv_PD2], mean=mean, cov=cov)

def plot_joint_default(upsilon1, upsilon2, sigma1, sigma2, r, rho, figStyle=None):
    # Grid of capital ratios
    k1_vals = np.linspace(0.07, 0.2, 50)
    k2_vals = np.linspace(0.07, 0.2, 50)
    K1, K2 = np.meshgrid(k1_vals, k2_vals)
    Z = np.zeros_like(K1)

    # Evaluate joint default probabilities
    for i in range(K1.shape[0]):
        for j in range(K1.shape[1]):
            Z[i, j] = joint_default_prob(K1[i, j], K2[i, j], upsilon1, upsilon2, sigma1, sigma2, r, rho)

    # Plotting
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    if figStyle == 'wireframe':
        ax.plot_wireframe(K1, K2, Z, rstride=1, cstride=1, color='steelblue', linewidth=0.6)
    else:
        surf = ax.plot_surface(K1, K2, Z, cmap='viridis', edgecolor='k', linewidth=0.1, antialiased=True)

    # Labels with LaTeX-style math and readable font sizes
    ax.set_xlabel(r'$k_1$', fontsize=11, labelpad=10)
    ax.set_ylabel(r'$k_2$', fontsize=11, labelpad=10)
    #ax.set_zlabel(r'$\Pr(L_1 = 1, L_2 = 1)$', fontsize=11, labelpad=10)
    #ax.set_title('Joint Probability of Default (JPD)', fontsize=12, pad=15)

    # Optional: color bar for surface
    #if figStyle != 'wireframe':
    #    fig.colorbar(surf, shrink=0.65, aspect=10, label='JPD')
    ax.set_zlim(0, 0.18)
    ax.view_init(elev=30, azim=30)  # Better viewing angle
    plt.tight_layout()
    #plt.show()
    saveFig(path,'JPD9') 


#%%
# Run the plot
plot_joint_default(upsilon1, upsilon2, sigma1, sigma2, .0, .9,'wireframe')

#%% Optimizer

def minimize_joint_default_given_kbar(k_bar, upsilon1, upsilon2, sigma1, sigma2, r, rho):
    """
    Minimize joint default probability subject to constraint:
        0.5 * k1 + 0.5 * k2 = k_bar
    """
    
    def objective(k):
        k1, k2 = k
        return joint_default_prob(k1, k2, upsilon1, upsilon2, sigma1, sigma2, r, rho)

    # Equality constraint: average capital ratio
    constraint = {'type': 'eq', 'fun': lambda k: 0.5 * k[0] + 0.5 * k[1] - k_bar}

    # Bounds to keep k1 and k2 within reasonable values (positive and less than 1/upsilon)
    k1_bounds = (0.07, 0.8)
    k2_bounds = (0.07, 0.8)
    bounds = [k1_bounds, k2_bounds]

    # Initial guess
    k0 = [0.01, 0.01]

    result = minimize(objective, k0, bounds=bounds, constraints=[constraint])

    if result.success:
        k1_opt, k2_opt = result.x
        jpd_opt = result.fun
        return k1_opt, k2_opt, jpd_opt
    else:
        raise ValueError("Optimization failed: " + result.message)




#%% 
k_bar = 0.10
k1_opt, k2_opt, jpd_opt = minimize_joint_default_given_kbar(
    k_bar, upsilon1, upsilon2, sigma1, sigma2, r, rho
)

print(f"Optimal k1 = {k1_opt:.4f}")
print(f"Optimal k2 = {k2_opt:.4f}")
print(f"Minimum JPD = {jpd_opt:.6f}")

#%%  Minimize JPD over a grid of values

def compute_optimal_k_and_jpd_vs_sigma1(k_bar, upsilon1, upsilon2, sigma1_vals, sigma2, r, rho, num_points=50):
    """
    Compute optimal k1, k2, and joint default probability (JPD) for a range of sigma1 values.
    """
    k1_opts = []
    k2_opts = []
    jpd_opts = []

    for sigma1 in sigma1_vals:
        try:
            k1, k2, jpd = minimize_joint_default_given_kbar(
                k_bar, upsilon1, upsilon2, sigma1, sigma2, r, rho
            )
            k1_opts.append(k1)
            k2_opts.append(k2)
            jpd_opts.append(jpd)
        except ValueError as e:
            print(f"Optimization failed at sigma1 = {sigma1:.4f}: {e}")
            k1_opts.append(np.nan)
            k2_opts.append(np.nan)
            jpd_opts.append(np.nan)

    return k1_opts, k2_opts, jpd_opts

def plot_optimal_k_and_jpd_vs_sigma1(sigma1_vals, k1_opts, k2_opts, jpd_opts, k_bar, sigma1_ref=None):
    """
    Plot optimal k1, k2, JPD, and individual default probabilities as functions of sigma1.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # First subplot: optimal capital ratios
    axs[0].plot(sigma1_vals, k1_opts, label=r'$k_1^*$', color='navy')
    axs[0].plot(sigma1_vals, k2_opts, label=r'$k_2^*$', color='darkgreen')
    axs[0].set_ylabel(r'Optimal Capital Ratios')
    axs[0].grid(True)
    axs[0].legend(loc='best')

    # Second subplot: joint default probability
    axs[1].plot(sigma1_vals, jpd_opts, label='Min JPD', color='crimson')
    axs[1].set_ylabel('Min JPD')
    axs[1].grid(True)

    # Third subplot: individual PDs
    pd1_vals = [PD(k, upsilon1, sigma, r) if not np.isnan(k) else np.nan
                for k, sigma in zip(k1_opts, sigma1_vals)]
    pd2_vals = [PD(k, upsilon2, sigma2, r) if not np.isnan(k) else np.nan
                for k in k2_opts]

    axs[2].plot(sigma1_vals, pd1_vals, label=r'PD$_1$', color='navy')
    axs[2].plot(sigma1_vals, pd2_vals, label=r'PD$_2$', color='darkgreen')
    axs[2].set_xlabel(r'$\sigma_1$')
    axs[2].set_ylabel('Default Probabilities')
    axs[2].grid(True)
    axs[2].legend(loc='best')

    # Add vertical line at reference sigma1 if provided
    if sigma1_ref is not None:
        for ax in axs:
            ax.axvline(x=sigma1_ref, color='black', linestyle='dotted')

    fig.suptitle(rf'Optimal Capital Allocation, JPD, and PDs vs. $\sigma_1$\n($\bar{{k}} = {k_bar}$)', fontsize=13)
    plt.tight_layout()
    plt.show()

#%% Parameters
num_points = 25
sigma1_vals = np.linspace(.03, 0.2, num_points)

k_bar = 0.15  # example value, adjust as needed
 
# Run optimization
k1_opts, k2_opts, jpd_opts = compute_optimal_k_and_jpd_vs_sigma1(
    k_bar, upsilon1, upsilon2, sigma1_vals, sigma2, r, rho, num_points
)

# Plot results
plot_optimal_k_and_jpd_vs_sigma1(sigma1_vals, k1_opts, k2_opts, jpd_opts, k_bar, sigma2)

#%% 

def plot_conditional_pd_vs_M(upsilon, sigma, r, rho, k_vals):
    M = np.linspace(-2.5, 2.5, 500)
    plt.figure(figsize=(4, 3))
    
    # Line styles for each PD level
    line_styles = ['-', '--', ':']
    colors = ['black', '#00bfc4', '#91c95c']  # Match your plot

    for idx, k in enumerate(k_vals):
        PD_k = PD(k, upsilon, sigma, r)
        inv_PD = norm.ppf(PD_k)

        PD_M = norm.cdf((inv_PD - rho * M) / np.sqrt(1 - rho**2))
        plt.plot(M, PD_M, linestyle=line_styles[idx], color=colors[idx], linewidth=2, label=f'$k_i$ = {k:.2f}')

    plt.xlabel('$M \\sim \\mathcal{N}(0,1)$', fontsize=12)
    plt.ylabel('$PD$', fontsize=12)
    plt.ylim(0, 1.)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.title('Conditional Probability of Default vs. $M$', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    saveFig(path,'PD_M_rho6') 


#%%    
plot_conditional_pd_vs_M(upsilon1, sigma1, r=.0, rho=.6, k_vals=[0.07, 0.15])

    

#%%

def plot_conditional_pd_surface(upsilon, sigma, r, M_val):
    """
    Plot PD(M) as a function of rho and k for a fixed M.
    """
    # Define grid
    rho_vals = np.linspace(0.0, 0.9, 50)
    k_vals = np.linspace(0.07, 0.2, 50)
    RHO, K = np.meshgrid(rho_vals, k_vals)
    PD_M_vals = np.zeros_like(RHO)

    # Fill in conditional PD values
    for i in range(RHO.shape[0]):
        for j in range(RHO.shape[1]):
            k = K[i, j]
            rho = RHO[i, j]
            pd = PD(k, upsilon, sigma, r)
            inv_pd = norm.ppf(pd)
            PD_M_vals[i, j] = norm.cdf((inv_pd - rho * M_val) / np.sqrt(1 - rho**2))

    # 3D Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, RHO, PD_M_vals, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$k$', fontsize=12)
    ax.set_ylabel(r'$\rho$', fontsize=12)
    ax.set_zlabel(r'$PD(M)$', fontsize=12)
    ax.set_title(rf'Conditional Default Probability vs. $k$ and $\rho$ (at $M$ = {M_val})', fontsize=13)
    #fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    #plt.show()

#%%
plot_conditional_pd_surface(upsilon1, sigma1, r, M_val=-2)  # You can try M_val = -1, 1, etc.
