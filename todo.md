# TODO / Future Improvements
- [ ] add readme.md & .gitignore (readme including diagram etc.)
- [ ] rewrite bias predictor pipeline to work with arbitrary calibration approaches with a .fit and .predict method
- [ ] rewrite preprocessing, to call from a list of common utility functions, and define preprocessing steps in the config file
- [ ] uncertainty associated with bias predictions for a given set of input features and domain of model applicability
- [ ] add more wake models and parameters in Pywake API & make analysis_pywake.yaml dynamic
- [ ] add tests
- [ ] add more wake models and parameters in Pywake API
- [ ] option to define your own layout features or call from available ones, and define this in config
- [ ] enable outputting of a fitted pipeline which can be directly called (without running the whole upstream workflow) on new datapoints

# Potential Issues / Things hardcoded
- assuming a specific form based on EDF dataset directory (base directory + case names)
- preprocessing steps will differ for other datasets. Could make a big list of common utility functions for preprocessing (based on prior datasets), continually develop, and define the required preprocessing steps in the config file.
- feature selection is hardcoded
- ML pipeline is defined in the main script. Try define in config.


