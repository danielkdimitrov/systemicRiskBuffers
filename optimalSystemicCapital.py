# -*- coding: utf-8 -*-
"""
Systemic Capital Allocation Model

Created on Tue Aug 23 16:55:02 2022
Author: Daniel
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution, LinearConstraint, Bounds
from myplotstyle import *


class PDmodel:
    def __init__(self, varyParam, paramsDict, useP2R=True, k_micro=0.07):
        """
        Main model to evaluate or optimize systemic capital allocation.

        Parameters:
            varyParam (str): Either 'min ES', 'evaluate ES', or 'Social Opt'
            paramsDict (dict): Parameters of the banking system
            useP2R (bool): Whether to include pillar 2 requirements
            k_micro (float): Microprudential capital floor
        """
        self.nSims = int(1e5)
        self.r = 0.0
        self.dict = {}
        self.k_micro = np.array([k_micro] * len(paramsDict['Names'])) + paramsDict['k_p2r']
        self.EAD = paramsDict['wts']
        self.LGD = paramsDict['LGD']
        self.rwa_intensity = paramsDict['rwa_intensity']
        self.Sigma_base = paramsDict['Sigma']
        self.k_bar = paramsDict['k_bar']
        self.sysIndex = np.ones(len(paramsDict['Names']), dtype=bool)
        self.Names = paramsDict['Names']
        self.nBanks = len(self.Names)
        self.nF = paramsDict['Rho'].shape[1]
        self.Lbar = paramsDict['Lbar']
        self.U_base = self.factorSim(paramsDict['Rho'])
        # simulate independent losses
        np.random.seed(42) 
        self.LGD_sim = np.random.triangular(left=0.3, mode=0.65, right=1.0, size=(self.nSims, self.nBanks))
        
        X_opt, self.dict["PD_init"] = self.DD(self.rwa_intensity, self.k_micro, self.Sigma_base, self.r)


        if varyParam == 'min ES':
            self.optimize_ES()

        elif varyParam == 'evaluate':
            self.evaluate(paramsDict['O-SII rates'])

        elif varyParam == 'Social Opt':
            self.Lambda = 0.18
            self.Eta = 0.024
            self.K_bar = np.linspace(0, 0.2, 20)
            self.paramsDict = paramsDict
            self.ECost, self.dfPDsys, self.dfES, self.SCB, self.k_bar_min = self.getECost()
        
        elif varyParam == 'min PDsys':
            self.optimize_PDsys_diffEv()
            
    def optimize_PDsys(self):
        """
        Optimize macroprudential capital allocation to minimize systemic PD (PDsys),
        using 'trust-constr' method with LinearConstraint for total capital budget.
        """
    
        n = len(self.EAD)
    
        def objective(k_macro):
            Xx, _ = self.DD(self.rwa_intensity, self.k_micro + k_macro, self.Sigma_base, self.r)
            IndD, _ = self.defaultSims(self.U_base, Xx)
            L = IndD.T * self.LGD_sim
            Lsys = np.sum(L * self.EAD, axis=1)
            return self.getPDsys(Lsys)
    
        # Constraint: total macro capital used ≤ k_bar
        A = self.EAD.reshape(1, -1)  # shape (1, n)
        capital_constraint = LinearConstraint(A, lb=-np.inf, ub=self.k_bar)
    
        # Bounds: each k_i ≥ 0
        bounds = Bounds(lb=np.zeros(n), ub=np.full(n, 0.5))  # max 20% buffer per bank (adjust if needed)
    
        # Better initial guess: distribute k_bar uniformly (normalized by EAD)
        x0 = np.full(n, self.k_bar / np.sum(self.EAD))
    
        # Baseline micro-only PDsys
        X_micro, _ = self.DD(self.rwa_intensity, self.k_micro, self.Sigma_base, self.r)
        IndD_micro, _ = self.defaultSims(self.U_base, X_micro)
        L_micro = IndD_micro.T * self.LGD_sim
        Lsys_micro = np.sum(L_micro * self.EAD.T, axis=1)
        self.dict["PDsysMicro"] = self.getPDsys(Lsys_micro)
        self.dict["ESmicro"], self.dict["MESmicro"] = self.getES(np.zeros(n), self.U_base, showMES=True)
    
        # Run optimizer
        result = minimize(
            objective,
            x0=x0,
            method='trust-constr',
            bounds=bounds,
            constraints=[capital_constraint],
            options={'verbose': 0, 'maxiter': 500}
        )
    
        # Store results
        k_macro_opt = result.x
        self.dict["k_macro_str"] = k_macro_opt
        self.dict["k_str"] = self.k_micro[self.sysIndex] + k_macro_opt
        self.dict["ESopt"], self.dict["MESopt"] = self.getES(k_macro_opt, self.U_base, showMES=True)
    
        X_final, self.dict["PD"] = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
        IndD_final, _ = self.defaultSims(self.U_base, X_final)
        L_final = IndD_final.T * self.LGD_sim
        Lsys_final = np.sum(L_final * self.EAD.T, axis=1)
        self.dict["PDsys"] = self.getPDsys(Lsys_final)
        self.Lsys = Lsys_final
        self.dict["k_bar_str"] = np.sum(self.dict["k_macro_str"]*self.EAD)

    def optimize_PDsys_diffEv(self):
        """
        Optimize macroprudential capital allocation to minimize systemic PD (PDsys),
        using differential evolution with a soft penalty for the capital constraint.
        """
    
        n = len(self.EAD)
    
        def objective(k_macro):
            # Capital constraint penalty
            capital_used = np.dot(k_macro, self.EAD)
            penalty = 1e3 * max(0, capital_used - self.k_bar) ** 2
    
            # Simulate PDsys
            Xx, _ = self.DD(self.rwa_intensity, self.k_micro + k_macro, self.Sigma_base, self.r)
            IndD, _ = self.defaultSims(self.U_base, Xx)
            #L = IndD.T * self.LGD
            # get simulated losses
            L = IndD.T * self.LGD_sim  # elementwise multiply: each default gets its own LGD
            Lsys = np.sum(L * self.EAD, axis=1)  # EAD is shape (n_banks,)
            #Lsys = np.sum(L * self.EAD.T, axis=1)
            pd_sys = self.getPDsys(Lsys)
    
            return pd_sys + penalty
    
        # Bounds: k_macro ≥ 0, with an upper cap (e.g., 0.2)
        bounds = [(0, .9)] * n
    
        # Baseline (micro-only)
        X_micro, _ = self.DD(self.rwa_intensity, self.k_micro, self.Sigma_base, self.r)
        IndD_micro, _ = self.defaultSims(self.U_base, X_micro)
        L_micro = IndD_micro.T * self.LGD_sim
        Lsys_micro = np.sum(L_micro * self.EAD, axis=1)
        self.dict["PDsysMicro"] = self.getPDsys(Lsys_micro)
        self.dict["ESmicro"], self.dict["MESmicro"] = self.getES(np.zeros(n), self.U_base, showMES=True)
    
        # Optimize using differential evolution
        sol = differential_evolution(
            objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=10,
            tol=1e-6,
            polish=True,
            #workers=-1,         # use all CPU cores
            updating='deferred'  # async update improves performance
            #disp=False
        )
    
        # Store results
        best_k_macro = sol.x
        self.dict["k_macro_str"] = best_k_macro
        self.dict["k_str"] = self.k_micro[self.sysIndex] + best_k_macro
        self.dict["ESopt"], self.dict["MESopt"] = self.getES(best_k_macro, self.U_base, showMES=True)
    
        X_final, self.dict["PD"] = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
        IndD_final, _ = self.defaultSims(self.U_base, X_final)
        L_final = IndD_final.T * self.LGD_sim
        Lsys_final = np.sum(L_final * self.EAD, axis=1)
        self.dict["PDsys"] = self.getPDsys(Lsys_final)
        self.Lsys = Lsys_final
        self.dict["k_bar_str"] = np.sum(self.dict["k_macro_str"]*self.EAD)

    def optimize_ES(self):
        """
        Solve for the macroprudential capital allocation that minimizes expected shortfall (ES)
        subject to a weighted capital constraint.
        """
        f = lambda k_macro: self.getES(k_macro, self.U_base)
        constraint = {'type': 'ineq', 'fun': lambda k_macro: self.k_bar - np.sum(k_macro * self.EAD)}
        bounds = [(0, None)] * len(self.EAD)
        sol = minimize(f, x0=np.zeros_like(self.EAD), method='SLSQP', bounds=bounds, constraints=[constraint])

        self.dict["k_macro_str"] = sol.x
        self.dict["k_str"] = self.k_micro[self.sysIndex] + sol.x
        self.dict["ESopt"], self.dict['MESopt'] = self.getES(sol.x, self.U_base, showMES=True)
        self.dict["ESmicro"], self.dict['MESmicro'] = self.getES(np.zeros_like(sol.x), self.U_base, showMES=True)

        Xx, self.dict["PD"] = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
        IndD, _ = self.defaultSims(self.U_base, Xx)
        L = IndD.T * self.LGD
        Lsys = np.sum(L * self.EAD.T, axis=1)
        self.dict["PDsys"] = self.getPDsys(Lsys)
        self.Lsys = Lsys

    def evaluate(self, k_macro_input):
        """
        Evaluate ES, systemic PD (PDsys), and individual bank default probabilities (PD)
        for a given macroprudential capital allocation.
    
        Stores results in self.dict:
            - k_macro_str: input O-SII rates
            - k_str: total capital buffer per bank
            - ESopt: expected shortfall under systemic distress
            - MESopt: marginal expected shortfall per bank
            - PD: individual default probabilities
            - PDsys: probability of systemic loss > Lbar
        """
        self.dict["k_macro_str"] = k_macro_input
        self.dict["k_str"] = self.k_micro[self.sysIndex] + k_macro_input
    
        # Expected Shortfall and MES
        self.dict["ESopt"], self.dict["MESopt"] = self.getES(k_macro_input, self.U_base, showMES=True)
    
        # ES with zero macroprudential buffer (micro-only benchmark)
        self.dict["ESmicro"], self.dict["MESmicro"] = self.getES(np.zeros_like(k_macro_input), self.U_base, showMES=True)
    
        # Default probabilities
        Xx, PD_individual = self.DD(self.rwa_intensity, self.dict["k_str"], self.Sigma_base, self.r)
        self.dict["PD_str"] = PD_individual
    
        # Simulate defaults and compute systemic PD
        IndD, _ = self.defaultSims(self.U_base, Xx)
        L = IndD.T * self.LGD
        Lsys = np.sum(L * self.EAD.T, axis=1)
        self.dict["PDsys"] = self.getPDsys(Lsys)
        self.Lsys = Lsys

    def DD(self, uspilon, k, sigma, r):
        DD = (-np.log(1. - uspilon * k) + (r - 0.5 * sigma ** 2)) / sigma
        X = -DD
        PD = np.clip(norm.cdf(X), 1e-10, 1.0)  # ⬅️ ensure no nan or exact 0
        return X, PD

    def factorSim(self, rho):
        np.random.seed(1)
        sims = np.random.normal(0.0, 1.0, size=(self.nF + self.nBanks, self.nSims))
        M, dZ = sims[:self.nF], sims[self.nF:]

        if self.nF == 1:
            return M * rho + np.sqrt(1 - rho ** 2) * dZ
        else:
            U = np.zeros((self.nBanks, self.nSims))
            for j in range(self.nBanks):
                rho_j = rho[j, :]
                U[j, :] = M.T @ rho_j + np.sqrt(1 - rho_j.T @ rho_j) * dZ[j, :]
            return U

    def defaultSims(self, U, Xx):
        IndD = np.zeros((self.nBanks, self.nSims))
        IndD[U <= Xx.reshape(self.nBanks, 1)] = 1
        return IndD, np.zeros(self.nSims)

    def getMES(self, IndD):
        L = IndD.T * self.LGD
        Lsys = np.sum(L * self.EAD.T, axis=1)
        MES = np.average(L[Lsys > self.Lbar], axis=0)
        PDsys = self.getPDsys(Lsys)
        return MES, PDsys

    def getES(self, k_macro, U, showMES=False):
        k_total = self.k_micro.copy()
        k_total[self.sysIndex] = k_total[self.sysIndex] + k_macro
        Xx, _ = self.DD(self.rwa_intensity, k_total, self.Sigma_base, self.r)
        IndD, _ = self.defaultSims(U, Xx)
        MES, PDsys = self.getMES(IndD)
        ESsys = np.sum(self.EAD * MES) * PDsys
        return (ESsys, MES) if showMES else PDsys

    def getPDsys(self, Lsys):
        return np.mean(Lsys > self.Lbar)

    def getECost(self):
        dfES = pd.DataFrame(index=self.K_bar, columns=["ES"])
        dfPDsys = pd.DataFrame(index=self.K_bar, columns=["PDsys"])
        dfKimacro = pd.DataFrame(index=self.K_bar, columns=self.Names)

        for k_bar in self.K_bar:
            self.paramsDict['k_bar'] = k_bar
            model = PDmodel('min ES', self.paramsDict)
            dfES.loc[k_bar] = model.dict['ESopt']
            dfKimacro.loc[k_bar] = model.dict['k_macro_str']
            dfPDsys.loc[k_bar] = model.dict['PDsys']

        dfFirstTerm = dfPDsys * self.Lambda * dfES.values
        SCB = self.Eta * (self.K_bar.reshape(-1, 1) - 0.0)
        dfSecond = (1 - dfPDsys) * SCBs
        ECost = dfFirstTerm + dfSecond
        min_idx = np.argmin(ECost.values)
        k_bar_min = self.K_bar[min_idx]
        return ECost, dfPDsys, dfES, SCB, k_bar_min


# --- Example usage ---

def getBaseParams():
    return {
        'Names': ['Bank 1', 'Bank 2'],
        'Sigma': np.array([0.1, 0.2]),
        'wts': np.array([0.5, 0.5]),
        'LGD': np.array([1.0, 1.0]),
        'Rho': np.array([[0.5]]),
        'rwa_intensity': np.array([0.3, 0.3]),
        'O-SII rates': np.array([0.05, 0.05]),
        'k_bar': 0.6,
        'Lbar': 0.1,
        'k_p2r': np.array([0.0, 0.0])
    }

def getRandomSigmaParams(n):
    """
    Generate synthetic parameters for n identical banks,
    differing only in asset volatility (sigma).
    """
    np.random.seed(42)  # Optional: for reproducibility

    sigmas = np.random.uniform(0.07, .11, size=n)

    return {
        'Names': [f'Bank {i+1}' for i in range(n)],
        'Sigma': sigmas,
        'wts': np.full(n, 1.0 / n),
        'LGD': np.ones(n),
        'Rho': np.array([[0.5]]),  # Diagonal=1.0, off-diagonal=0.5
        'rwa_intensity': np.full(n, 0.3),
        'O-SII rates': np.full(n, 0.07),
        'k_bar': 0.6,
        'Lbar': 1/n,
        'k_p2r': np.zeros(n)
    }

def main():
    print("Running example: minimize PDsys")
    params = getBaseParams()
    model = PDmodel('min PDsys', params)
    print("Baseline PDsys (micro-pru only):", model.dict["PDsysMicro"])
    print("Optimized PDsys:", model.dict["PDsys"])
    
    print("Optimal k_macro (to minimize PDsys):", model.dict["k_macro_str"])
    print("Expected Shortfall:", model.dict["ESopt"])
    print("Individual PDs:", model.dict["PDsys"])



if __name__ == "__main__":
    main()
