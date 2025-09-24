try:
    import umbra
except:
    print('Umbra not installed.')


from ..postprocesser import PostProcesser


class BayesianCalibration(PostProcesser):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
    



if __name__ == "__main__":
    bc = BayesianCalibration()
    print(bc)
