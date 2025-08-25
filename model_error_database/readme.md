# Contents:
- EDF_datasets: contains the reference data from the different cases. There is a meta.yaml file in each folder manually configured to make running the simulations easier. There is also an analysis_pywake.yaml file in each folder configured currently with the Bastankhah2014 model
- results_stacked_hh.nc: This is the database of model errors with the reference physical inputs as hub height values
- results_stacked_hh_best_sample.nc: Same as above but following some ad-hoc "best" parameter selection and addition of more layout features.
- results_stacked_fullprofile.nc: database of model errors but containing the full vertical profile physical inputs also. There is also another coordinate for heights normalized by the hub height for the given wind farm. This is not currently being used in the predictor but may be used to create features to predict the bias based on vertical profile.

# Scripts:
## preprocessing_physical_inputs: 
- Ensure all cases have the same height dimensions by interpolation
- Update boundary layer parameters (BLH, capping inversion etc.) based on temperature profile using ci_fitting function in WAYVE API
- Update turbulence intensity based on TKE and wind speed
- Optionally Plotting the difference between old and updated input parameters
- Create a new netcdf file with physical inputs


## run_pywake_sweep: 
- takes in the model parameters we are using as samples, the windIO data for running the pywake api and 
- outputs a database of model bias (%) for a particular case. 
- Model bias values are in (%) (average error/average power*100)

## main.py: 
- calls run_pywake_sweep
- combines the datasets from each case into a single dataset.
- There are 2 datasets. One where physical inputs are interpolated to hub height and one where they are not. "results_stacked.nc" and "results_stacked_heights.nc". Heights are normalized by the hub height in the latter
- finds the best combination of parameters in terms of miniminimizing the Mean Squared Bias.
- Adds some new layout features and outputs a new data set with only the best parameter combination.

## error_predictor:
- Input database
- Preprocessing data, changing to a Pandas dataframe format,removing some columns and splitting features and target
- Model training function where different pipelines are configured
- Comparing cross validation scores from different model pipelines

# To:Do
- Perhaps there are extra features that calculate things based on the vertical profile at the rotor which can be added in
