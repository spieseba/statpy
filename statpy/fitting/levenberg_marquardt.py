import numpy as np

# default Levenberg-Marquardt parameter
lm_parameter = {
    "delta": 1e-5,
    "lmbd0": 1e-2,
    "maxiter": None,
    "eps1": 1e-3,
    "eps2": 1e-3,
    "eps3": 1e-1,
    "eps4": 1e-1,
    "Lup": 11.0,
    "Ldown": 9.0,
    "update_type": 3
}

class LevenbergMarquardt:
    """
    Levenberg-Marquardt fitter based on https://people.duke.edu/~hpgavin/ce281/lm.pdf

        Parameters:
        -----------
                data (tuple): (t numpy array, y numpy array, stds numpy array) data points (ti, yi, stdi).
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                chi_square (function): chi squared function which takes data and model parameters as an input.
                p0 (numpy array): array containing the start values of the parameter vector.
                lm_parameter (dict, optional): dictionary containing the parameters for the Levenberg-Marquardt minimization algorithm.
                                C (2D numpy array): 2D array containing the covariance matrix of the data if known. Default is None.
                                delta (float): fractional increment of p for numerical derivatives. Default is 1e-3.
                                lmbd0 (float): damping parameter which determines whether Levenberg-Marquardt update resembles gradient descent or Gauss-Newton update. Default is 1e-2.
                                maxiter (int): maximum number of iterations. Default is 10 * N_parameter
                                eps1 (float): convergence tolerance for gradient: max|J^T W (y - y_hat)| < eps1. Default is 1e-3.
                                eps2 (float): convergence tolerance for parameters: max|h_i / p_i| < eps2. Default is 1e-3.
                                eps3 (float): convergence tolerance for reduced chi2: chi^2/(N_data - N_parameter + 1) < eps3. Default is 1e-1.
                                eps4 (float): acceptance of a L-M step. Default us 1e-1.
                                Lup (float): factor for increasing lambda. Default is 11.
                                Ldown (float): factor for decreasing lambda. Default is 9.
                                update_type (int): Determines update method of L-M-algorithm. Default is 3.
                                                    1: Levenberg-Marquardt lambda update
                                                    2: Quadratic update
                                                    3: Nielsen's lambda update equations
        Returns:
        --------
                p (numpy array): estimate of the parameter values
                chi2 (float): chi squared     
                i (int): number of iterations
                converged (boolean): Success of fitting
    """
    def __init__(self, t, y, W, model, p0, lm_parameter_user={}):
        self.t = t; self.y = y; self.W = W
        self.model = model
        self.p0 = p0
        self.Np = len(p0)
        # change default minimization parameters with user specified input
        for param, value in lm_parameter_user.items():
            lm_parameter[param] = value
        self.delta = lm_parameter["delta"]
        self.lmbd0 = lm_parameter["lmbd0"]
        if lm_parameter["maxiter"] == None:
            self.maxiter = 10 * len(self.p0)
        else:
            self.maxiter = lm_parameter["maxiter"]
        # convergence criteria
        self.eps1 = lm_parameter["eps1"]
        self.eps2 = lm_parameter["eps2"]
        self.eps3 = lm_parameter["eps3"]
        # acceptance of L-M step
        self.eps4 = lm_parameter["eps4"]
        # factors for chaning lambda
        self.Lup = lm_parameter["Lup"]
        self.Ldown = lm_parameter["Lup"]
        # update method
        self.update_type = lm_parameter["update_type"]

    def __call__(self):
        update_types = {1: self.LevenbergMarquardtUpdate, 2: self.QuadraticUpdate, 3:self.NielsenLambdaUpdate}
        lambda_inits = {1: self.lambda_init1, 2: self.lambda_init2, 3: self.lambda_init2}

        p = self.p0
        update = update_types[self.update_type]
        J = jacobian(self.model, self.t, p, self.delta)()
        lmbd = lambda_inits[self.update_type](self.lmbd0, J)
        nu = 2
        for i in range(self.maxiter):
            J = jacobian(self.model, self.t, p, self.delta)()
            converged, p, lmbd, chi2, nu = update(p, lmbd, J, nu)
            if converged:
                return p, chi2, i, True, J
        return p, chi2, i, False, J

    def chi_squared(self, p):
        return (self.model(self.t, p) - self.y) @ self.W @ (self.model(self.t, p) - self.y)

    def lambda_init1(self, lmbd0, J):
        return lmbd0

    def lambda_init2(self, lmbd0, J):
        return lmbd0 * np.max(np.diag(np.transpose(J) @ self.W @ J))

    def LevenbergMarquardtUpdate(self, p, lmbd, J, nu):
        yp = self.model(self.t, p)
        chi2 = self.chi_squared(p)
        b = np.transpose(J) @ self.W @ (self.y - yp)

        # check criterion 1
        if np.max(np.abs(b)) < self.eps1:
            return True, p, None, chi2, None
        # check criterion 3
        if chi2 / (self.Np - len(p) + 1) < self.eps3:
            return True, p, None, chi2, None

        # use eq.13 to determine h
        B = np.transpose(J) @ self.W @ J
        A = B + lmbd * np.diag(np.diag(B))
        h = np.linalg.solve(A, b)

        # use eq.16 for metric rho
        chi2h = self.chi_squared(p+h)
        rho = (chi2 - chi2h) / (np.transpose(h) @ ( lmbd * np.diag(np.diag(B)) @ h + b ))
        if rho > self.eps4: 
            p = p + h
            lmbd = max(lmbd/self.Ldown, 10e-7)
        else:
            lmbd = min(lmbd*self.Lup, 10e7)
        
        # check criterion 2
        if(np.max(np.abs(h / p))) < self.eps2:
            return True, p, None, chi2, None
        return False, p, lmbd, chi2, None

    def QuadraticUpdate(self, p, lmbd, J, nu):
        yp = self.model(self.t, p)
        chi2 = self.chi_squared(p)
        b = np.transpose(J) @ self.W @ (self.y - yp)

        # check criterion 1
        if np.max(np.abs(b)) < self.eps1:
            return True, p, None, chi2, None
        # check criterion 3
        if chi2 / (self.Np - len(p) + 1) < self.eps3:
            return True, p, None, chi2, None

        # use eq.12 to determine h
        B = np.transpose(J) @ self.W @ J
        A = B + lmbd * np.diag(np.ones_like(np.diag(B)))
        h = np.linalg.solve(A, b)

        # determine alpha and use eq.15 for metric rho
        chi2h = self.chi_squared(p+h)
        alpha = (b @ h) / ( (chi2h - chi2)/2 + 2 * b @ h)
        chi2ah = self.chi_squared(p + (alpha * h))
        rho = (chi2 - chi2ah) / (np.transpose(alpha * h) @ ( lmbd * alpha * h + b ))
        if rho > self.eps4: 
            p = p + alpha * h
            lmbd = max(lmbd/(1 + alpha), 10e-7)
        else:
            lmbd = lmbd + np.abs(chi2ah - chi2) / (2*alpha) 

        # check criterion 2
        if np.max(np.abs(h / p)) < self.eps2:
            return True, p, None, chi2, None
        return False, p, lmbd, chi2, None

    def NielsenLambdaUpdate(self, p, lmbd, J, nu):
        yp = self.model(self.t, p)
        chi2 = self.chi_squared(p)
        b = np.transpose(J) @ self.W @ (self.y - yp)

        # check criterion 1
        if np.max(np.abs(b)) < self.eps1:
            return True, p, None, chi2, None
        # check criterion 3
        if chi2 / (self.Np - len(p) + 1) < self.eps3:
            return True, p, None, chi2, None

        # use eq.12 to determine h
        B = np.transpose(J) @ self.W @ J
        A = B + lmbd * np.diag(np.ones_like(np.diag(B)))
        h = np.linalg.solve(A, b)

        # eq.15 for metric rho
        chi2h = self.chi_squared(p+h)
        rho = (chi2 - chi2h) / (np.transpose(h) @ ( lmbd * h + b ))
        if rho > self.eps4: 
            p = p + h
            lmbd = lmbd * max(1.0/3.0, 1.0 - (2.0*rho - 1)**3)
            nu = 2
        else:
            lmbd = lmbd * nu
            nu = 2 * nu 
        
        # check criterion 2
        if np.max(np.abs(h / p)) < self.eps2:
            return True, p, None, chi2, None 
        return False, p, lmbd, chi2, nu


class Jacobian:
    def __init__(self, model, t, p, delta):
        self.model = model
        self.t = t
        self.p = p
        self.delta = delta

    def __call__(self, method="central difference"):
        if method == "central difference":
            return np.array([[self.central_diff(ti, j) for j in range(len(self.p))] for ti in self.t])
        else:
            raise Exception("Method not implemented")
    
    def central_diff(self, ti, j):
        Delta = np.zeros_like(self.p)
        Delta[j] = self.delta
        return (self.model(ti, self.p+Delta) - self.model(ti, self.p-Delta)) / (2 * self.delta)

def param_cov_lm(J, W):
    return np.linalg.inv(np.transpose(J) @ W @ J)

def param_std_err_lm(J, W):
    return np.sqrt( np.diag( param_cov_lm(J, W) ) )

def fit_std_err_lm(J, cov_p):
    return np.sqrt( np.diag(J @ cov_p @ np.transpose(J)) )
