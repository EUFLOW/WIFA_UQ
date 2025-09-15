Model calibration and bias prediction for an arbitrary validation dataset adhering to the windIO format. Includes modularity in terms of data preprocessing requirements, wake model selection, calibration method and bias prediction model

# Contents:
- EDF_datasets: contains the reference data from the different cases. There is a meta.yaml file in each folder manually configured to make running the simulations easier. There is also an analysis_pywake.yaml file in each folder configured currently with the Bastankhah2014 model
- results_stacked_hh.nc: This is the database of model errors with the reference physical inputs as hub height values

# Scripts:
## run_pywake_sweep: 
- takes in the model parameters we are using as samples, the windIO data for running the pywake api and 
- outputs a database of model bias (%) for a particular case. 
- Model bias values are in (%) (average error/average power*100)

## preprocessing: 
- Ensure all cases have the same height dimensions by interpolation
- Update boundary layer parameters (BLH, capping inversion etc.) based on temperature profile using ci_fitting function in WAYVE API
- Update turbulence intensity based on TKE and wind speed
- Optionally Plotting the difference between old and updated input parameters
- Create a new netcdf file with physical inputs

## database_gen.py: 
- calls run_pywake_sweep
- combines the datasets from each case into a single dataset.
- Adds some additional layout features.

## error_predictor:
- Input database and machine learning pipeline object
- Carries out cross validation to calibrate and train model

## main
- navigate to validation data
- define parameter samples for database
- define wake model, calibration approach and ML pipeline


# To Do:
In terms of modularity...


For now most options are hardcoded, e.g., there is only one approach to model calibration, one wake model (one pywake configuration) implemented. In future, Several model calibration classes could be available in the error_predictor workflow. run_pywake_sweep could be repeated or refactored for another wake model. the preprocessing script could be altered for a new dataset
