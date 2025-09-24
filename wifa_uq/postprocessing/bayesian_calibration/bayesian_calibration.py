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

        self.flow_model = umbra.flow_model.WIFAModel(open(system_yaml))
        print(self.flow_model.name_model)

        self.__initialize_bayesian_model()


    def __initialize_bayesian_model(self):
        # Initalize prior
        params = []
        dists = []
        for param, range in self.params.items():
            assert (len(range) == 2) and (range[0]<range[1])
            params.append(param)
            dists.append(umbra.distributions.Uniform(range[0], range[1]))
        prior = umbra.bayesian_model.Prior(params, dists)
        # Initalize likelihood
        likelihood = umbra.bayesian_model.LikelihoodTurbinePower(flow_model=self.flow_model, data=self.data)
        # Initialize bayesian model
        self.bayesian_model = umbra.bayesian_model.BayesianModel(prior, likelihood, name='umbra4wifa')

    def fit(self):
        sampler = umbra.sampler.TMCMC(self.bayesian_model, parallel=False)
        inference_data = sampler.sample()
        return inference_data.samples


    def predict(self):
        pass
    

