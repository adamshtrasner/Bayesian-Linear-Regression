import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from copy import deepcopy


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        return np.vander(x/degree, degree+1, increasing=True)
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        H = np.concatenate([np.exp(-((x[:, None] - c)**2)/(2*(beta**2))) for c in centers], axis=1)
        intercept = np.ones(H.shape[0])[:, None]
        return np.concatenate((intercept, H), axis=1)
    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        vonder_3_deg = np.vander(x, 4, increasing=True)
        H = np.concatenate([np.maximum(x[:, None] - k, 0)**3 for k in knots], axis=1)
        return np.concatenate((vonder_3_deg, H), axis=1)
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.estimators)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions
        self.posterior_mean = None
        self.posterior_cov = None
        self.posterior = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        M = self.sig * np.identity(y.shape[0]) + H @ self.theta_cov @ H.T
        term = self.theta_cov @ H.T @ np.linalg.inv(M)

        self.posterior_mean = self.theta_mean + term @ (y - H @ self.theta_mean)
        self.posterior_cov = self.theta_cov - term @ H @ self.theta_cov

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        y_pred = self.basis_functions(X) @ self.posterior_mean
        return y_pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        std_pred = np.sqrt(np.diagonal(H @ self.posterior_cov @ H.T) + self.sig)
        return std_pred

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        posterior_sample = np.random.multivariate_normal(self.posterior_mean, self.posterior_cov)
        y_pred = self.basis_functions(X) @ posterior_sample
        return y_pred


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions
        self.estimators = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        H_dagger = np.linalg.pinv(H)
        self.estimators = H_dagger @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        y_pred = self.basis_functions(X) @ self.estimators
        return y_pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16)//2]
    train_hours = nov16_hours[:len(nov16)//2]
    test = nov16[len(nov16)//2:]
    test_hours = nov16_hours[len(nov16)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        pred = ln.predict(test_hours)
        ase = np.mean((test - pred)**2)
        print(f'Average squared error with LR and d={d} is {ase:.2f}')

        # plot graphs for linear regression part
        plt.figure()
        tr = plt.scatter(train_hours, train)
        ts = plt.scatter(test_hours, test)
        plt.legend((tr, ts),
                   ('train', 'test'),
                   loc='upper left',
                   fontsize=8)
        plt.plot(test_hours, pred, lw=2, color="black")
        plt.title(f'Temperatures on Second Half of November 16 2020\n d={d}, ASE={ase:.2f}')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

    # ----------------------------------------- Bayesian Linear Regression
    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 3  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([2, 4, 8, 12, 16, 20, 22])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)
        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        # plot prior graphs

        # find mean function
        H = pbf(x)
        mean = H @ mu
        std = np.sqrt(np.diagonal(H@cov@H.T) + sigma)

        plt.figure()
        plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')
        for i in range(5):
            prior_sample = np.random.multivariate_normal(mu, cov)
            plt.plot(x, H@prior_sample, lw=2)

        plt.plot(x, mean, lw=2, label='prior mean')
        plt.legend()
        plt.title('Daily Average Temperatures in November')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)

        # print average squared error performance
        mmse_pred = blr.predict(test_hours)
        std_pred = blr.predict_std(test_hours)
        ase = np.mean((test - mmse_pred) ** 2)
        print(f'Average squared error with BLR and d={deg} is {ase:.2f}')

        plt.figure()
        tr = plt.scatter(train_hours, train)
        ts = plt.scatter(test_hours, test)

        plt.fill_between(test_hours, mmse_pred - std_pred, mmse_pred + std_pred,
                         alpha=.5, label='confidence interval')
        for i in range(5):
            posterior_sample = blr.posterior_sample(test_hours)
            plt.plot(test_hours, posterior_sample, lw=2)

        plt.plot(test_hours, mmse_pred, lw=2, color="black", label="MMSE pred")
        plt.legend()
        plt.title(f'Temperatures on Second Half of November 16 2020\n d={deg}, ASE={ase:.2f}')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

    # # ---------------------- Gaussian basis functions

    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        H = rbf(x)
        mean = H @ mu
        std = np.sqrt(np.diagonal(H @ cov @ H.T) + sigma)

        plt.figure()
        plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')
        for i in range(5):
            prior_sample = np.random.multivariate_normal(mu, cov)
            plt.plot(x, H @ prior_sample, lw=2)

        plt.plot(x, mean, lw=2, label='prior mean')
        plt.legend()
        plt.title('Daily Average Temperatures in November')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)

        # print average squared error performance
        mmse_pred = blr.predict(test_hours)
        std_pred = blr.predict_std(test_hours)
        ase = np.mean((test - mmse_pred) ** 2)
        print(f'Average squared error with BLR and center=S{ind+1} is {ase:.2f}')

        plt.figure()
        tr = plt.scatter(train_hours, train)
        ts = plt.scatter(test_hours, test)

        plt.fill_between(test_hours, mmse_pred - std_pred, mmse_pred + std_pred,
                         alpha=.5, label='confidence interval')
        for i in range(5):
            posterior_sample = blr.posterior_sample(test_hours)
            plt.plot(test_hours, posterior_sample, lw=2)

        plt.plot(test_hours, mmse_pred, lw=2, color="black", label="MMSE pred")
        plt.legend()
        plt.title(f'Temperatures on Second Half of November 16 2020\n center=S{ind+1}, ASE={ase:.2f}')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        csbf = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, csbf)

        blr = BayesianLinearRegression(mu, cov, sigma, csbf)

        # plot prior graphs
        H = csbf(x)
        mean = H @ mu
        std = np.sqrt(np.diagonal(H @ cov @ H.T) + sigma)

        plt.figure()
        plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')
        for i in range(5):
            prior_sample = np.random.multivariate_normal(mu, cov)
            plt.plot(x, H @ prior_sample, lw=2)

        plt.plot(x, mean, lw=2, label='prior mean')
        plt.legend()
        plt.title('Daily Average Temperatures in November')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()

        # plot posterior graphs
        blr.fit(train_hours, train)

        # print average squared error performance
        mmse_pred = blr.predict(test_hours)
        std_pred = blr.predict_std(test_hours)
        ase = np.mean((test - mmse_pred) ** 2)
        print(f'Average squared error with BLR and knot=K{ind + 1} is {ase:.2f}')

        plt.figure()
        tr = plt.scatter(train_hours, train)
        ts = plt.scatter(test_hours, test)

        plt.fill_between(test_hours, mmse_pred - std_pred, mmse_pred + std_pred,
                         alpha=.5, label='confidence interval')
        for i in range(5):
            posterior_sample = blr.posterior_sample(test_hours)
            plt.plot(test_hours, posterior_sample, lw=2)

        plt.plot(test_hours, mmse_pred, lw=2, color="black", label="MMSE pred")
        plt.legend()
        plt.title(f'Temperatures on Second Half of November 16 2020\n knot=K{ind + 1}, ASE={ase:.2f}')
        plt.xlabel('hour')
        plt.ylabel('temperature [C]')
        plt.show()


if __name__ == '__main__':
    main()
