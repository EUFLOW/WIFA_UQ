

import os
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from shutil import rmtree

# File and output paths
nc_path = os.path.join(
    os.path.dirname(__file__),
    'results', 'rmse', 'calibration_results.nc'
)
output_dir = Path(os.path.dirname(__file__)) / 'results' / 'rmse' / 'figures' / 'linear_reg'
if output_dir.is_dir():
    rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

target = 'best_k'
excl = [target, "best_k_error", "best_k_P", "wind_direction", "time_index"]
feature_vars = ["turbulence_intensity", "wind_speed", "k", "capping_inversion_strength", "z0", "n_turbines"]
# Load dataset
with xr.open_dataset(nc_path) as ds:
    # Find candidate features (exclude best_k and non-numeric)
    all_vars = list(ds.variables)
    #feature_vars = [v for v in all_vars if v not in excl and ds[v].dtype.kind in 'fi']
    if len(feature_vars) < 1:
        raise ValueError('Not enough numeric features for regression.')

    y = ds[target].values.flatten()
    X = np.stack([ds[f].values.flatten() for f in feature_vars], axis=1)
    # Remove NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_ = X[mask]
    y_ = y[mask]
    if len(y_) < 2:
        raise ValueError('Not enough valid data for regression.')
    # Check rank of feature matrix
    rank = np.linalg.matrix_rank(X_)
    if rank < X_.shape[1]:
        print(f"Feature matrix is rank deficient (rank={rank}, features={X_.shape[1]}). Reducing rank using SVD.")
        U, S, Vt = np.linalg.svd(X_, full_matrices=False)
        X_reduced = U[:, :rank] @ np.diag(S[:rank])
        X_ = X_reduced
        feature_labels = [f'SVD_{i+1}' for i in range(rank)]
    else:
        feature_labels = feature_vars
    # Fit linear regression
    model = Ridge()#LinearRegression()
    model.fit(X_, y_)
    y_pred = model.predict(X_)
    # Estimate prediction uncertainty using cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(model, X_, y_, cv=kf)
    residuals = y_ - y_cv_pred
    std_resid = np.std(residuals)
    # Plot true vs predicted with uncertainty
    plt.figure(figsize=(7, 7))
    plt.scatter(y_, y_pred, alpha=0.7, label='Prediction')
    # Plot 95% prediction interval
    y_line = np.linspace(y_.min(), y_.max(), 100)
    plt.plot(y_line, y_line, 'r--', label='Ideal fit')
    plt.fill_between(y_line, y_line - 1.96*std_resid, y_line + 1.96*std_resid, color='gray', alpha=0.2, label='95% Prediction Interval')
    plt.xlabel('True best_k')
    plt.ylabel('Predicted best_k')
    if len(feature_labels) == len(feature_vars):
        plt.title(f'Ridge Regression: best_k from all features')
    else:
        plt.title(f'Ridge Regression: best_k from SVD components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'true_vs_pred_best_k_all_features.png')
    plt.close()
    # Plot all coefficients
    plt.figure(figsize=(6, max(4, len(feature_labels) * 0.5)))
    plt.barh(feature_labels, model.coef_)
    plt.xlabel('Coefficient (Slope)')
    if len(feature_labels) == len(feature_vars):
        plt.title('Linear Regression Coefficients (All Features)')
    else:
        plt.title('Linear Regression Coefficients (SVD Components)')
    plt.tight_layout()
    plt.savefig(output_dir / 'coefficients_all_features.png')
    plt.close()
print(f"Plots saved to {output_dir}")
