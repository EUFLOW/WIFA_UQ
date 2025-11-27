import os
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import panel as pn
pn.extension('plotly')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import RobustScaler, StandardScaler
from itertools import combinations
from shutil import rmtree

# File and output paths
nc_path = os.path.join(
    os.path.dirname(__file__),
    'results', 'rmse', 'calibration_results.nc'
)
output_dir = Path(os.path.dirname(__file__)) / 'results' / 'rmse' / 'figures' / 'gaussproc_reg'
if output_dir.is_dir():
    rmtree(output_dir)

nu = 2.5
allow_svd = False
max_error = 0.04

targets = ['best_k', 'best_k_error']#, 'best_k_farm_P']
#feature_vars = ["turbulence_intensity"]
#feature_vars = ["turbulence_intensity", "wind_speed"]
#feature_vars = ["turbulence_intensity", "wind_speed", "k", "capping_inversion_strength", "z0", "n_turbines"]
#feature_vars = ["ABL_height", "wind_speed", "k", "capping_inversion_thickness"]#, "z0", "n_turbines"]
#feature_vars = ["turbulence_intensity", "ABLHH"]
feature_vars = ["turbulence_intensity", "ABLHH", "z0", "n_turbines"]
#feature_vars = ["turbulence_intensity", 'ABLHH', "capping_inversion_thickabl","z0", "k", "n_turbines"]

# Load dataset
with xr.open_dataset(nc_path) as ds:
    # Find candidate features (exclude target and non-numeric)
    cs = ds["case"].values[ds["best_k_error"].values <= max_error]
    ds = ds.sel(case=cs)
    all_vars = list(ds.variables)
    if len(feature_vars) < 1:
        raise ValueError('Not enough numeric features for regression.')
    
    def read_data(name):
        if name == "farmPowerNormalized":
            return ds['best_k_farm_P'].values.flatten() / ds["capacity"].values.flatten()
        elif name == 'ABLHH':
            return ds['ABL_height'].values.flatten() / ds["hub_height"].values.flatten()
        elif name == 'hz0':
            return ds["hub_height"].values.flatten() / ds["z0"].values.flatten()
        elif name == "capping_inversion_thickabl":
            return ds["capping_inversion_thickness"].values.flatten() / ds["ABL_height"].values.flatten()
        else:
            return ds[name].values.flatten()

    for target in targets:
        y = read_data(target)
        odir = output_dir / target
        odir.mkdir(parents=True, exist_ok=True)

        scaler = StandardScaler() #RobustScaler()
        X = np.stack([read_data(f) for f in feature_vars], axis=1)
        # Remove NaNs
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_ = X[mask]
        y_ = y[mask]
        if len(y_) < 2:
            raise ValueError('Not enough valid data for regression.')
        # Rescale features
        # Check rank of feature matrix
        rank = np.linalg.matrix_rank(X_)
        if rank < X_.shape[1]:
            if not allow_svd:
                raise ValueError(f"Feature matrix is rank deficient (rank={rank}, features={X_.shape[1]}). Set allow_svd=True to reduce rank using SVD.")
            print(f"Feature matrix is rank deficient (rank={rank}, features={X_.shape[1]}). Reducing rank using SVD.")
            # SVD decomposition
            U, S, Vt = np.linalg.svd(X_, full_matrices=False)
            # Keep only the first 'rank' components
            X_reduced = U[:, :rank] @ np.diag(S[:rank])
            X_ = X_reduced
        X_scaled = scaler.fit_transform(X_)
        # Fit Gaussian Process Regression with Matern + WhiteKernel
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=nu) + WhiteKernel(noise_level=1e-7)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0, normalize_y=True)        
        # Squared exponential kernel (RBF)
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e3))
        #model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)
        # Typical kernel and parameters: Constant * RBF, default alpha, no normalization
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e3))
        #model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)
        model.fit(X_scaled, y_)
        y_pred, y_std = model.predict(X_scaled, return_std=True)
        # Plot true vs predicted with uncertainty
        plt.figure(figsize=(7, 7))
        plt.scatter(y_, y_pred, alpha=0.7, label='Prediction')
        # Plot 95% prediction interval (model std)
        plt.errorbar(y_, y_pred, yerr=1.96*y_std, fmt='none', ecolor='gray', alpha=0.3, label='95% Prediction Interval')
        y_line = np.linspace(y_.min(), y_.max(), 100)
        fpath = odir / f'{target}_true_vs_pred_best_k_all_features.png'
        print(f"Writing file {fpath}")
        plt.plot(y_line, y_line, 'r--', label='Ideal fit')
        plt.xlabel(f'True {target}')
        plt.ylabel(f'Predicted {target}')
        if X_.shape[1] == len(feature_vars):
            plt.title(f'Gaussian Process Regression: {target} from all features')
        else:
            plt.title(f'Gaussian Process Regression: {target} from SVD components')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()

        # Plot feature importances (length scales)
        if hasattr(model.kernel_, 'k2') and hasattr(model.kernel_.k2, 'length_scale'):
            plt.figure(figsize=(6, max(4, X_.shape[1] * 0.5)))
            if X_.shape[1] == len(feature_vars):
                plt.barh(feature_vars, model.kernel_.k2.length_scale)
                plt.xlabel('Length Scale (Feature Relevance)')
                plt.title('GP Feature Relevance (RBF Length Scales)')
            else:
                plt.barh([f'SVD_{i+1}' for i in range(X_.shape[1])], model.kernel_.k2.length_scale)
                plt.xlabel('Length Scale (SVD Component)')
                plt.title('GP Feature Relevance (SVD Components)')
            plt.tight_layout()
            plt.savefig(odir / 'feature_relevance_all_features.png')
            plt.close()

        # Visualize dependence on all feature_vars (only if not rank reduced)
        if X_.shape[1] == len(feature_vars):
            X_median = np.median(X_, axis=0)
            for i, var in enumerate(feature_vars):
                if var == "k":
                    var = "RANS_k"
                var_range = np.linspace(np.percentile(X_[:, i], 1), np.percentile(X_[:, i], 99), 100)
                X_pred = np.tile(X_median, (len(var_range), 1))
                X_pred[:, i] = var_range
                X_pred_scaled = scaler.transform(X_pred)
                y_pred_var, y_std_var = model.predict(X_pred_scaled, return_std=True)
                fpath = odir / f'{target}_dependence_on_{var}.png'
                print(f"Writing file {fpath}")
                plt.figure(figsize=(8, 5))
                plt.plot(var_range, y_pred_var, label='Prediction', color='b')
                plt.fill_between(var_range, y_pred_var - 1.96*y_std_var, y_pred_var + 1.96*y_std_var, color='b', alpha=0.2, label='95% Confidence Interval')
                plt.xlabel(var)
                plt.ylabel(f'Predicted {target}')
                plt.title(f'Dependence of {target} on {var} (others fixed at median)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()
        else:
            print("Feature matrix was rank reduced; skipping feature-specific dependency plots.")


        # 2D surface plots for all pairs of features or SVD components
        X_median = np.median(X_, axis=0)
        num_axes = X_.shape[1]
        plotly_figs = []
        if num_axes == len(feature_vars):
            axis_labels = feature_vars
        else:
            axis_labels = [f'SVD_{i+1}' for i in range(num_axes)]
        for i, j in combinations(range(num_axes), 2):

            var1, var2 = axis_labels[i], axis_labels[j]
            if var1 == "k":
                var1 = "RANS_k"
            if var2 == "k":
                var2 = "RANS_k"
            v1_range = np.linspace(np.percentile(X_[:, i], 1), np.percentile(X_[:, i], 99), 40)
            v2_range = np.linspace(np.percentile(X_[:, j], 1), np.percentile(X_[:, j], 99), 40)
            V1, V2 = np.meshgrid(v1_range, v2_range)
            X_pred = np.tile(X_median, (V1.size, 1))
            X_pred[:, i] = V1.ravel()
            X_pred[:, j] = V2.ravel()
            X_pred_scaled = scaler.transform(X_pred)
            y_pred_2d, y_std_2d = model.predict(X_pred_scaled, return_std=True)
            Y_pred = y_pred_2d.reshape(V1.shape)
            Y_std = y_std_2d.reshape(V1.shape)

            # Plot mean prediction surface
            fpath = odir / f'{target}_dependence_on_{var1}_and_{var2}.png'
            print(f"Writing file {fpath}")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(V1, V2, Y_pred, cmap='viridis', alpha=0.8)
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel(f'Predicted {target}')
            if num_axes == len(feature_vars):
                ax.set_title(f'Dependence of {target} on {var1} and {var2}\n(others fixed at median)')
            else:
                ax.set_title(f'Dependence of {target} on {var1} and {var2} (SVD components)')
            fig.colorbar(surf, shrink=0.5, aspect=10, label=f'Predicted {target}')
            plt.tight_layout()
            plt.savefig(fpath)
            plt.close()

            # Plot relative uncertainty surface
            fpath = odir / f'{target}_relative_uncertainty_on_{var1}_and_{var2}.png'
            print(f"Writing file {fpath}")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            rel_uncertainty = np.divide(Y_std, np.abs(Y_pred) + 1e-8)  # avoid division by zero
            surf = ax.plot_surface(V1, V2, rel_uncertainty, cmap='plasma', alpha=0.8)
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel('Relative Uncertainty')
            if num_axes == len(feature_vars):
                ax.set_title(f'Relative Uncertainty of {target} on {var1} and {var2}\n(others fixed at median)')
            else:
                ax.set_title(f'Relative Uncertainty of {target} on {var1} and {var2} (SVD components)')
            fig.colorbar(surf, shrink=0.5, aspect=10, label=f'Std/|Mean| of {target}')
            plt.tight_layout()
            plt.savefig(fpath)
            plt.close()

            # --- Plotly interactive surfaces ---
            # Prediction surface
            title_pred = f"{target}: {var1} vs {var2} (Prediction)"
            fig_pred = go.Figure(data=[go.Surface(z=Y_pred, x=V1, y=V2, colorscale='Viridis')])
            fig_pred.update_layout(title=title_pred, scene=dict(
                xaxis_title=var1, yaxis_title=var2, zaxis_title=f'Predicted {target}'
            ))
            plotly_figs.append((title_pred, fig_pred))
            # Relative uncertainty surface
            title_unc = f"{target}: {var1} vs {var2} (Relative Uncertainty)"
            fig_unc = go.Figure(data=[go.Surface(z=rel_uncertainty, x=V1, y=V2, colorscale='Plasma')])
            fig_unc.update_layout(title=title_unc, scene=dict(
                xaxis_title=var1, yaxis_title=var2, zaxis_title='Relative Uncertainty'
            ))
            plotly_figs.append((title_unc, fig_unc))

        # --- Save dashboard with all interactive 3D plots after all targets processed ---
        if len(plotly_figs) > 0:
            # Group prediction and uncertainty plots side by side for each variable pair
            dashboard_rows = []
            for idx in range(0, len(plotly_figs), 2):
                pred_title, pred_fig = plotly_figs[idx]
                unc_title, unc_fig = plotly_figs[idx+1] if idx+1 < len(plotly_figs) else (None, None)
                row = pn.Row(
                    pn.pane.Plotly(pred_fig, sizing_mode='stretch_both', min_height=800, name=pred_title),
                    pn.pane.Plotly(unc_fig, sizing_mode='stretch_both', min_height=800, name=unc_title)
                )
                dashboard_rows.append(row)
            dashboard = pn.Column(*dashboard_rows)
            dash_path = odir / f'{target}_gaussproc_3d_dashboard.html'
            print(f"\nSaving interactive dashboard to {dash_path}\n")
            pn.panel(dashboard).save(str(dash_path), embed=True)

