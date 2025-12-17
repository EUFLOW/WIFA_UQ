# TODO / Future Improvements
- [x] add readme.md & .gitignore (readme including diagram etc.)
- [ ] Implement an example of probabilistic calibration
- [x] rewrite preprocessing, to call from a list of common utility functions, and define preprocessing steps in the config file
- [ ] uncertainty associated with bias predictions for a given set of input features and domain of model applicability
- [ ] add more wake models and parameters in Pywake API & make analysis_pywake.yaml dynamic
- [ ] add tests
- [ ] add more wake models and parameters in Pywake API
- [ ] option to define your own layout features or call from available ones, and define this in config
- [x] enable outputting of a fitted pipeline which can be directly called (without running the whole upstream workflow) on new datapoints

# Potential Issues / Things hardcoded
- ML pipeline is defined in the main script. Try define in config.
