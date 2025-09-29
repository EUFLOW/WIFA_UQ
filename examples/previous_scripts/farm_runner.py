import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from wifa.pywake_api import run_pywake
from windIO.utils.yml_utils import load_yaml

# Define the configurations
spacings = ['DX5D', 'DX7D', 'DX9D']
turbine_counts = [4, 25, 100]
blending_values = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0

# Create results directories
os.makedirs('farm_results', exist_ok=True)
os.makedirs('farm_figures', exist_ok=True)

# Dictionary to store results
results = {}
all_power_diffs = []

# Process each farm configuration
for dx in spacings:
    for num_turbines in turbine_counts:
        print(f"Processing {dx} with {num_turbines} turbines")
        
        # Define paths
        config_key = f"{dx}_DY5D_Turbine_NumberNumber{num_turbines}"
        system_file = f'farm_simulations/windio_Farm_ABL_IEA15/wind_energy_system/system_staggered_{config_key}.yaml'
        ref_data_file = f'farm_simulations/Result_code_saturne_Farm_calibration_data_LIGHT/farm_result/turbine_data_staggered_{config_key}.nc'
        
        # Load reference data
        ref_data = xr.load_dataset(ref_data_file)
        
        # Extract numerical spacing value
        dx_value = float(dx[2:-1])  # Extracts '5' from 'DX5D'
        
        # Load system configuration
        system_config = load_yaml(system_file)
        
        # Run with Linear superposition model
        linear_config = system_config.copy()
        linear_config['attributes']['analysis']['superposition_model']['ws_superposition'] = 'Linear'
        output_dir_linear = f'farm_results/{dx}_{num_turbines}_linear'
        print("  Running simulation with Linear superposition")
        run_pywake(linear_config, output_dir=output_dir_linear)
        
        # Run with Squared superposition model
        squared_config = system_config.copy()
        squared_config['attributes']['analysis']['superposition_model']['ws_superposition'] = 'Squared'
        output_dir_squared = f'farm_results/{dx}_{num_turbines}_squared'
        print("  Running simulation with Squared superposition")
        run_pywake(squared_config, output_dir=output_dir_squared)
        
        # Load results from both simulations
        linear_res = xr.load_dataset(f'{output_dir_linear}/PowerTable.nc')
        squared_res = xr.load_dataset(f'{output_dir_squared}/PowerTable.nc')
        
        power_diffs = []
        
        # Test different blending parameters
        for blend_idx, blend in enumerate(blending_values):
            print(f"  Testing blending parameter {blend:.1f}")
            
            # Blend the power outputs using the blending parameter
            # blend = 0 means pure linear, blend = 1 means pure squared
            blended_power = (1 - blend) * linear_res.power + blend * squared_res.power
            
            # Calculate power difference (RMSE)
            power_diff = np.sqrt(((ref_data.power - blended_power) ** 2).mean(['turbine', 'time']))
            power_diffs.append(float(power_diff))
            
            # Store detailed information for analysis
            all_power_diffs.append({
                'spacing_dx': dx_value,
                'num_turbines': num_turbines,
                'blend': blend,
                'power_diff': float(power_diff),
                'config': f"{dx}_T{num_turbines}"
            })
            
            # Create visualization for this blend
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot power for each turbine (averaged over time)
            mean_ref_power = ref_data.power.mean('time')
            mean_blended_power = blended_power.mean('time')
            
            ax.plot(range(len(mean_ref_power)), mean_ref_power, 'o-', label='Reference')
            ax.plot(range(len(mean_blended_power)), mean_blended_power, 'x-', label='Blended Simulation')
            
            ax.set_xlabel('Turbine Index')
            ax.set_ylabel('Power (W)')
            ax.set_title(f"{dx}, {num_turbines} turbines, blend={blend:.1f}, RMSE={power_diff:.2e}")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f'farm_figures/{dx}_{num_turbines}_blend_{blend:.1f}_power.png')
            plt.close()
        
        # Find optimal blending value
        optimal_blend_idx = np.argmin(power_diffs)
        optimal_blend = blending_values[optimal_blend_idx]
        
        # Store results
        results[(dx, num_turbines)] = {
            'optimal_blend': optimal_blend,
            'power_diffs': power_diffs,
            'min_power_diff': power_diffs[optimal_blend_idx],
            'dx_value': dx_value
        }
        
        # Plot power difference vs. blending parameter
        plt.figure(figsize=(8, 5))
        plt.plot(blending_values, power_diffs, 'o-')
        plt.axvline(optimal_blend, color='r', linestyle='--', 
                   label=f'Optimal blend = {optimal_blend:.2f}')
        plt.xlabel('Blending Parameter')
        plt.ylabel('Power Difference (RMSE)')
        plt.title(f'Power Difference vs. Blending Parameter\n{dx}, {num_turbines} turbines')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'farm_figures/{dx}_{num_turbines}_blend_optimization.png')
        plt.close()

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(all_power_diffs)

# Create a comprehensive plot showing optimal blending vs. turbine count and spacing
fig, ax = plt.subplots(figsize=(10, 6))

for dx in spacings:
    dx_value = float(dx[2:-1])
    optimal_blends = [results[(dx, n)]['optimal_blend'] for n in turbine_counts]
    ax.plot(turbine_counts, optimal_blends, 'o-', label=f'{dx} ({dx_value}D spacing)')

ax.set_xlabel('Number of Turbines')
ax.set_ylabel('Optimal Blending Parameter')
ax.set_title('Optimal Superposition Blending vs. Turbine Count and Spacing')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('farm_figures/optimal_blend_vs_turbines_and_spacing.png')

# Modeling the relationship between farm characteristics and optimal blending
model_data = []
for dx in spacings:
    dx_value = float(dx[2:-1])
    for num_turbines in turbine_counts:
        model_data.append({
            'spacing_dx': dx_value,
            'num_turbines': num_turbines,
            'optimal_blend': results[(dx, num_turbines)]['optimal_blend']
        })

model_df = pd.DataFrame(model_data)

# Create regression model
X = model_df[['spacing_dx', 'num_turbines']]
y = model_df['optimal_blend']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Print model coefficients
print("\nRegression Model:")
print(f"optimal_blend = {model.coef_[0]:.4f} * spacing_dx + {model.coef_[1]:.6f} * num_turbines + {model.intercept_:.4f}")

# Create prediction plot
# Create a mesh grid of spacing_dx and num_turbines
spacing_range = np.linspace(min(model_df['spacing_dx']), max(model_df['spacing_dx']), 20)
turbine_range = np.linspace(min(model_df['num_turbines']), max(model_df['num_turbines']), 20)
spacing_grid, turbine_grid = np.meshgrid(spacing_range, turbine_range)

# Predict for all combinations
predictions = np.zeros(spacing_grid.shape)
for i in range(spacing_grid.shape[0]):
    for j in range(spacing_grid.shape[1]):
        predictions[i, j] = model.predict([[spacing_grid[i, j], turbine_grid[i, j]]])[0]

# Plot 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(spacing_grid, turbine_grid, predictions, cmap='viridis', alpha=0.8)

# Plot actual data points
for idx, row in model_df.iterrows():
    ax.scatter(row['spacing_dx'], row['num_turbines'], row['optimal_blend'], 
               color='red', s=50, edgecolor='k')

ax.set_xlabel('Turbine Spacing (X direction, diameters)')
ax.set_ylabel('Number of Turbines')
ax.set_zlabel('Optimal Blending Parameter')
ax.set_title('Model of Optimal Blending Parameter')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('farm_figures/optimal_blending_model_3d.png')

# Additional analysis: Create a heatmap of optimal blending values
plt.figure(figsize=(10, 6))
heatmap_data = np.zeros((len(spacings), len(turbine_counts)))

for i, dx in enumerate(spacings):
    for j, num_turbines in enumerate(turbine_counts):
        heatmap_data[i, j] = results[(dx, num_turbines)]['optimal_blend']

plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(label='Optimal Blending Parameter')
plt.xticks(range(len(turbine_counts)), turbine_counts)
plt.yticks(range(len(spacings)), [f"{s} ({float(s[2:-1])}D)" for s in spacings])
plt.xlabel('Number of Turbines')
plt.ylabel('Turbine Spacing')
plt.title('Optimal Superposition Blending Parameters')
plt.tight_layout()
plt.savefig('farm_figures/optimal_blending_heatmap.png')

# Save results to files
results_df.to_csv('farm_results/all_simulation_results.csv', index=False)
model_df.to_csv('farm_results/optimal_blending_model_data.csv', index=False)

print("\nAnalysis completed. Results saved to farm_results/ and farm_figures/ directories.")
