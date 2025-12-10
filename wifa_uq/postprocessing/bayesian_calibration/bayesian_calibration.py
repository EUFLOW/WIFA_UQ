from wifa_uq.postprocessing.postprocesser import PostProcesser
import numpy as np
from wifa_uq.postprocessing.calibration import (
    MinBiasCalibrator,
    #  DefaultParams,
    # LocalParameterPredictor,
)

try:
    import umbra
except Exception as e:
    print("Umbra not installed.")
    print(e)
    quit()


class BayesianCalibration(PostProcesser):
    def __init__(self, system_yaml, params, data):
        self.system_yaml = system_yaml
        self.params = params
        self.data = data

        self._flow_model = umbra.flow_model.WIFAModel(open(system_yaml))
        self._initialize_bayesian_model()

        self._samples_posterior = None
        self._samples_posterior_predictive = None
        self._inference_data = None

    def _initialize_bayesian_model(self):
        # Initalize prior
        params = []
        dists = []
        for param, range in self.params.items():
            assert (len(range) == 2) and (range[0] < range[1])
            params.append(param)
            dists.append(umbra.distributions.Uniform(range[0], range[1]))
        prior = umbra.bayesian_model.Prior(params, dists)
        # Initalize likelihood
        likelihood = umbra.bayesian_model.LikelihoodTurbinePower(
            flow_model=self._flow_model, data=self.data
        )
        # Initialize bayesian model
        self._bayesian_model = umbra.bayesian_model.BayesianModel(
            prior, likelihood, name="umbra4wifa"
        )

    def fit(self):
        sampler = umbra.sampler.TMCMC(self._bayesian_model, parallel=False)
        self._inference_data = sampler.sample()
        self._samples_posterior = self._inference_data.samples
        return self._samples_posterior

    def predict(self):
        self._samples_posterior_predictive = self._bayesian_model.posterior_predictive(
            self.samples_posterior
        )
        return self._samples_posterior_predictive

    @property
    def samples_posterior(self):
        return self._samples_posterior

    @property
    def samples_posterior_predictive(self):
        return self._samples_posterior_predictive


class BayesianCalibrationWrapper:
    """
    Wrapper to make BayesianCalibration compatible with MainPipeline.

    This performs Bayesian inference to get a posterior distribution
    of parameters, then uses the MAP (Maximum A Posteriori) estimate
    as the calibrated parameters.
    """

    def __init__(
        self, dataset_train, system_yaml: str = None, param_ranges: dict = None
    ):
        """
        Args:
            dataset_train: xarray Dataset with the model bias database
            system_yaml: Path to windIO system YAML (required for WIFA model)
            param_ranges: Dict of {param_name: [min, max]} for prior bounds
        """
        self.dataset_train = dataset_train
        self.system_yaml = system_yaml
        self.param_ranges = param_ranges or {}

        self.swept_params = dataset_train.attrs.get("swept_params", [])
        self.best_idx_ = None
        self.best_params_ = None
        self.posterior_samples_ = None

        # Build param_ranges from database if not provided
        if not self.param_ranges:
            self._infer_param_ranges()

    def _infer_param_ranges(self):
        """Infer parameter ranges from database coordinates."""
        for param_name in self.swept_params:
            if param_name in self.dataset_train.coords:
                values = self.dataset_train.coords[param_name].values
                self.param_ranges[param_name] = [
                    float(values.min()),
                    float(values.max()),
                ]

    def fit(self):
        """
        Run Bayesian inference and extract MAP estimate.

        Note: This requires UMBRA to be installed and a valid system_yaml.
        Falls back to MinBiasCalibrator if UMBRA is unavailable.
        """
        try:
            if self.system_yaml is None:
                raise ValueError("system_yaml path required for Bayesian calibration")

            # Prepare observation data from dataset
            # This is simplified - you may need to adjust based on your data structure
            ref_power = self.dataset_train["ref_power_cap"].isel(sample=0).values

            # Create BayesianCalibration instance
            bc = BayesianCalibration(
                system_yaml=self.system_yaml, params=self.param_ranges, data=ref_power
            )

            # Run inference
            self.posterior_samples_ = bc.fit()

            # Extract MAP estimate (mode of posterior)
            self.best_params_ = {}
            for i, param_name in enumerate(self.swept_params):
                samples = self.posterior_samples_[:, i]
                # Use median as robust point estimate
                self.best_params_[param_name] = float(np.median(samples))

            # Find closest sample index to MAP estimate
            self.best_idx_ = self._find_closest_sample_idx()

        except ImportError:
            print("WARNING: UMBRA not installed. Falling back to MinBiasCalibrator.")
            fallback = MinBiasCalibrator(self.dataset_train)
            fallback.fit()
            self.best_idx_ = fallback.best_idx_
            self.best_params_ = fallback.best_params_

        except Exception as e:
            print(
                f"WARNING: Bayesian calibration failed ({e}). Falling back to MinBiasCalibrator."
            )
            fallback = MinBiasCalibrator(self.dataset_train)
            fallback.fit()
            self.best_idx_ = fallback.best_idx_
            self.best_params_ = fallback.best_params_

        return self

    def _find_closest_sample_idx(self):
        """Find sample index closest to MAP estimate."""
        n_samples = len(self.dataset_train.sample)
        distances = np.zeros(n_samples)

        for param_name, map_value in self.best_params_.items():
            if param_name in self.dataset_train.coords:
                sample_values = self.dataset_train.coords[param_name].values
                distances += (sample_values - map_value) ** 2

        return int(np.argmin(distances))

    def get_posterior_samples(self):
        """Return posterior samples for uncertainty quantification."""
        return self.posterior_samples_
