from wifa_uq.postprocessing.postprocesser import PostProcesser

try:
    import umbra
except:
    print('Umbra not installed.')


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
            assert (len(range) == 2) and (range[0]<range[1])
            params.append(param)
            dists.append(umbra.distributions.Uniform(range[0], range[1]))
        prior = umbra.bayesian_model.Prior(params, dists)
        # Initalize likelihood
        likelihood = umbra.bayesian_model.LikelihoodTurbinePower(flow_model=self._flow_model, data=self.data)
        # Initialize bayesian model
        self._bayesian_model = umbra.bayesian_model.BayesianModel(prior, likelihood, name='umbra4wifa')

    def fit(self):
        sampler = umbra.sampler.TMCMC(self._bayesian_model, parallel=False)
        self._inference_data = sampler.sample()
        self._samples_posterior = self._inference_data.samples
        return self._samples_posterior

    def predict(self):
        self._samples_posterior_predictive = self._bayesian_model.posterior_predictive(self.samples_posterior)
        return self._samples_posterior_predictive
    
    @property
    def samples_posterior(self):
        return self._samples_posterior

    @property
    def samples_posterior_predictive(self):
        return self._samples_posterior_predictive