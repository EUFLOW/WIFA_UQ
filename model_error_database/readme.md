# Contents:
- EDF_datasets: contains the reference data from the different cases. There is a meta.yaml file in each folder manually configured to make running the simulations easier. There is also an analysis_pywake.yaml file in each folder configured currently with the turbopark model
- results_stacked_hh.nc: This is the database of model errors with the reference physical inputs as hub height values
- results_stacked_fullprofile: database of model errors but containing the full vertical profile physical inputs. There is also another coordinate for heights normalized by the hub height for the given wind farm
- scripts and notebooks described below

# Notebooks.
## preprocessing_physical_inputs: 
- optionally look at and print the physical inputs from any particular wind farm folder. 
- Ensure all cases have the same height dimensions by interpolation
- Update boundary layer parameters (BLH, capping inversion etc.) based on temperature profile using ci_fitting function in WAYVE API
- Update turbulence intensity based on TKE and wind speed
- Create a new netcdf file with physical inputs
	
## Checking_outputs:
- After creating the database... we can inspect the dataset and plot the power error vs flow case

## database_EDA:
- looking at the distribution of power error in the dataset vs flow case, and vs individual features. Also looking at correlation between features.

## error_predictor:
- Currently uses hub height physical data
- converts netcdf to a dataframe, with all variables in columns
- tunes xgboost parameters 
- Calculating shap scores in a cross validation loop to investigate the contribution of different features
- Manually selecting certain features based on interpretation of shap plots
- rerunning xgboost with new features and looking at cross validation results for some different metrics

# Scripts:
## run_pywake_sweep: 
- takes in the model parameters we are using as samples, the windIO data for running the pywake api and 
- outputs a database of model errors for a particular case. 
- Model errors are a RMSE value over each wind farm, and are normalized by the rated power for comparison between different wind farms. (effectively in units of capacity factor)

## main.py: 
- combines the datasets from each case into a single dataset.
- There are 2 datasets. One where physical inputs are interpolated to hub height and one where they are not. "results_stacked.nc" and "results_stacked_heights.nc". Heights are normalized by the hub height in the latter

# Gaps:
- physical inputs remain from the windlab, not recalculated from temperature profile (including ti)
- First version.... database creation is takes quite long and is not yet parallelized or memory efficient
