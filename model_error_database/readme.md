# Contents:
- EDF_datasets: contains the reference data from the different cases. There is a meta.yaml file in each folder manually configured to make running the simulations easier. There is also an analysis_pywake.yaml file in each folder configured currently with the Bastankhah2014 model
- results_stacked_hh.nc: This is the database of model errors with the reference physical inputs as hub height values
- results_stacked_fullprofile: database of model errors but containing the full vertical profile physical inputs also. There is also another coordinate for heights normalized by the hub height for the given wind farm. This is not currently being used in the predictor.
- scripts and notebooks described below

# Notebooks.
## preprocessing_physical_inputs: 
- optionally look at and print the physical inputs from any particular wind farm folder. 
- Ensure all cases have the same height dimensions by interpolation
- Update boundary layer parameters (BLH, capping inversion etc.) based on temperature profile using ci_fitting function in WAYVE API
- Update turbulence intensity based on TKE and wind speed
- Plotting the difference between old and updated input parameters
- Create a new netcdf file with physical inputs
	
## db_output_analysis:
- After creating the database... we can inspect the dataset and plot the power bias vs flow case
- There is also logic here to calculate layout specific features for each simulation (different wind farms and directions) and append them to the database for use in the predictor

## database_EDA:
- looking at the distribution of power bias in the dataset vs flow case, and vs individual features. 
- Also looking at correlation between features and target.
- Calculating SHAP feature importance scores and visualizing

# Scripts:
## run_pywake_sweep: 
- takes in the model parameters we are using as samples, the windIO data for running the pywake api and 
- outputs a database of model bias (%) for a particular case. 
- Model bias values are in (%) (average error/average power*100)

## main.py: 
- combines the datasets from each case into a single dataset.
- There are 2 datasets. One where physical inputs are interpolated to hub height and one where they are not. "results_stacked.nc" and "results_stacked_heights.nc". Heights are normalized by the hub height in the latter

## error_predictor:
- Currently uses hub height physical data
- Preprocessing data, changing to a Pandas dataframe format and splitting features and target
- Dimension reduction (currently this is via manual feature selection)
- Training the model using a pipeline object
- Cross val scores from both linear regression and xgboost pipeline objects are stored in csvs

The aim is to create an automated process from this, but for now the steps are:
- Run 1-preprocessing_physical_inputs.ipynb
- Run 2-main.py
- Run db_output_analysis.ipynb:
- Look at db_EDA.ipynb to identify features to exclude
- Enter features in error_predictor.py
