import numpy as np
from .levenberg_marquardt import LevenbergMarquardt

class fit:
    def __init__(self, data, model, minimizer):
        self.data = data
        self.model = model
        self.opt, self.opt_params = minimizer # minimizer should be tuple: (minimizer class, parameters)
        # self.p0 = p0 # include in opt_params 

    def chi_square(self, data, parameter):
        return sum([((self.model(x, parameter) - y) / err)**2.0 for x, y, err in data])

    def estimate_parameter(self, f, data):
        opt_res = self.opt.minimize(lambda p: self.chi_square(data, p), self.opt_params)
