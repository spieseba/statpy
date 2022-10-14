import numpy as np

class jacobian:
    def __init__(self, model, t, p, eps):
        self.model = model
        self.t = t
        self.p = p
        self.eps = eps

    def __call__(self, method="central difference"):
        if method == "central difference":
            return np.array([[self.central_diff(ti, j) for j in range(len(self.p))] for ti in self.t])
        else:
            raise Exception("Method not implemented")
    
    def central_diff(self, ti, j):
        Eps = np.zeros_like(self.p)
        Eps[j] = self.eps
        return (self.model(ti, self.p+Eps) - self.model(ti, self.p-Eps)) / (2 * self.eps)


class LevenbergMarquardt:
    def __init__(self):
        pass