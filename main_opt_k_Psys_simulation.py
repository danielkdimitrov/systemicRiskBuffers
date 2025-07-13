import matplotlib.pyplot as plt
import numpy as np
from optimalSystemicCapital import PDmodel, getBaseParams, getRandomSigmaParams


'''
# Optimize PDsys at the Base Case 
params = getBaseParams()
model_opt = PDmodel('min PDsys', params)

print("Optimal k_macro (to minimize PDsys):", model_opt.dict["k_macro_str"])
print('Optimal K_bar:',np.average(model_opt.dict["k_macro_str"]), ' vs. budgeted:',  model_opt.k_bar)
print("Systemic PD (optimized):", model_opt.dict["PDsys"])
print("Systemic PD (micro-only):", model_opt.dict["PDsysMicro"])

# Evaluate PDsys over a grid of uniform k_macro values
PDsys_vals = []
k_vals = np.linspace(0.0, 0.1, 20)

for α in k_vals:
    test_params = getBaseParams()
    test_params['O-SII rates'] = np.array([α, α])
    model = PDmodel('evaluate', test_params)
    PDsys_vals.append(model.dict["PDsys"])

# Plotting
plt.plot(k_vals, PDsys_vals, label='PDsys vs. uniform k_macro')
plt.axhline(model_opt.dict["PDsys"], color='r', linestyle='--', label='Minimized PDsys')
plt.axhline(model_opt.dict["PD_init"][0], color='grey', linestyle='--', label='Initial PD1')
plt.axhline(model_opt.dict["PD_init"][1], color='black', linestyle='--', label='Initial PD2')

plt.xlabel("k_macro (per bank)")
plt.ylabel("PDsys")
plt.title("Systemic PD vs Uniform Macroprudential Capital")
plt.legend()
plt.grid(True)
plt.show()
'''

n_banks = 5  # Set number of banks here
params = getRandomSigmaParams(n_banks)

# Grid over capital constraint
kbar_vals = np.linspace(0.01, 0.4, 15)

# --- Preallocate results ---
PDsys_vals = []
PDsys_micro_vals = []
k_macro_list = []
PD_init_list = []
PD_str_list = []

# --- Loop over k_bar values ---
for kbar in kbar_vals:
    print(f"\nRunning for k_bar = {kbar:.3f}")
    #overwrite kbar
    params['k_bar'] = kbar

    model = PDmodel('min PDsys', params)

    # Store values
    PDsys_vals.append(model.dict["PDsys"])
    PDsys_micro_vals.append(model.dict["PDsysMicro"])
    k_macro_list.append(model.dict["k_macro_str"])
    PD_init_list.append(model.dict["PD_init"])
    PD_str_list.append(model.dict["PD"])

    print("Optimal k_macro:", np.round(model.dict["k_macro_str"], 4))
    print("Systemic PD:", model.dict["PDsys"])
    print("Systemic PD (micro-only):", model.dict["PDsysMicro"])
    print('Optimal K_bar:',model.dict["k_bar_str"], ' vs. budgeted:',  model.k_bar)

# --- Convert lists to arrays ---
k_macro_arr = np.array(k_macro_list)       # shape (len(kbar_vals), n_banks)
PD_init_arr = np.array(PD_init_list)       # shape (len(kbar_vals), n_banks)
PD_str_arr = np.array(PD_str_list)         # shape (len(kbar_vals), n_banks)
PDsys_vals = np.array(PDsys_vals)
PDsys_micro_vals = np.array(PDsys_micro_vals)

# --- Plot 1: k_macro_str per bank ---
plt.figure(figsize=(10, 6))
for i in range(n_banks):
    plt.plot(kbar_vals, k_macro_arr[:, i], label=f'k_macro[{i}]')
plt.xlabel('Total k_bar')
plt.ylabel('Optimal k_macro allocation')
plt.title('Optimal Macroprudential Capital Allocation vs. k_bar')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: PDsys and PDsysMicro ---
plt.figure(figsize=(8, 5))
plt.plot(kbar_vals, PDsys_vals, label='PDsys (optimized)', color='blue')
plt.plot(kbar_vals, PDsys_micro_vals, label='PDsys (micro-only)', color='red', linestyle='--')
plt.xlabel('Total k_bar')
plt.ylabel('Systemic PD')
plt.title('Systemic PD vs. Capital Constraint')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Individual PDs before and after ---
plt.figure(figsize=(10, 6))
for i in range(n_banks):
    plt.plot(kbar_vals, PD_init_arr[:, i], linestyle='--', color='gray', alpha=0.5)
    plt.plot(kbar_vals, PD_str_arr[:, i], label=f'PD[{i}] (optimized)')
plt.xlabel('Total k_bar')
plt.ylabel('Individual PDs')
plt.title('Per-bank PDs vs. Capital Constraint')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


'''
def plot_PDsys_surface(model):
    grid = model.grid_results
    k1, k2 = np.meshgrid(grid['k1_vals'], grid['k2_vals'], indexing='ij')
    PDsys = grid['PDsys']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k1, k2, PDsys, cmap='viridis')
    ax.set_xlabel('k_macro[0]')
    ax.set_ylabel('k_macro[1]')
    ax.set_zlabel('PDsys')
    ax.set_title('Systemic PD Surface')
    plt.show()

def plot_individual_PD_lines(model):
    grid = model.grid_results
    k_vals = grid['k1_vals']

    plt.plot(k_vals, grid['PD_i'][:, 0], label='PD_1')
    plt.plot(k_vals, grid['PD_j'][0, :], label='PD_2')
    plt.xlabel('Capital Buffer (k_macro)')
    plt.ylabel('Default Probability')
    plt.title('Individual Bank PDs vs Capital')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_PDsys_heatmap(model):
    grid = model.grid_results
    k1_vals = grid['k1_vals']
    k2_vals = grid['k2_vals']
    PDsys = grid['PDsys']

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(k1_vals, k2_vals, PDsys.T, levels=20, cmap='Reds')  # transpose for correct orientation
    plt.colorbar(c, ax=ax, label='Systemic PD')
    ax.set_xlabel('k_macro[0]')
    ax.set_ylabel('k_macro[1]')
    ax.set_title('Heatmap of Systemic PD')
    plt.grid(True)
    plt.show()


def plot_individual_PD_heatmaps(model):
    grid = model.grid_results
    k1_vals = grid['k1_vals']
    k2_vals = grid['k2_vals']
    PD_i = grid['PD_i']
    PD_j = grid['PD_j']

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    c1 = axs[0].contourf(k1_vals, k2_vals, PD_i.T, levels=20, cmap='magma')
    plt.colorbar(c1, ax=axs[0], label='PD Bank 1')
    axs[0].set_xlabel('k_macro[0]')
    axs[0].set_ylabel('k_macro[1]')
    axs[0].set_title('Heatmap of PD Bank 1')

    c2 = axs[1].contourf(k1_vals, k2_vals, PD_j.T, levels=20, cmap='plasma')
    plt.colorbar(c2, ax=axs[1], label='PD Bank 2')
    axs[1].set_xlabel('k_macro[0]')
    axs[1].set_ylabel('k_macro[1]')
    axs[1].set_title('Heatmap of PD Bank 2')

    plt.tight_layout()
    plt.show()    

params = getBaseParams()
model = PDmodel('min PDsys', params)

plot_PDsys_surface(model)
plot_individual_PD_lines(model)
    
plot_PDsys_heatmap(model)
plot_individual_PD_heatmaps(model)

'''
#%%%



'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
k_i = 0.07              # Example value for k_i
upsilon_i = 0.3        # Example value for upsilon_i

# Create a grid of sigma values
sigma_grid = np.linspace(0.01, .5, 200)

# Compute the derivative over the grid
derivative_values = []
for sigma_i in sigma_grid:
    # Argument of phi
    x = (np.log(1 - upsilon_i * k_i) / sigma_i) + (0.5 * sigma_i)
    
    # Normal density function
    phi_x = norm.pdf(x)
    
    # Derivative expression
    dPD_dki = - (upsilon_i / (sigma_i * (1 - upsilon_i * k_i))) * phi_x
    derivative_values.append(dPD_dki)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(sigma_grid, derivative_values, label='dPD/dk_i vs sigma_i', color='dodgerblue')
plt.xlabel('σ (sigma_i)')
plt.ylabel('Derivative of PD with respect to k_i')
plt.title('Sensitivity of PD(k_i) to k_i over σ Grid')
plt.grid(True)
plt.legend()
plt.show()

'''