from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    print("noise level {}".format(noise))
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    y_no_noise = (x+3) * (x+2) * (x+1) * (x-1) * (x-2)
    y = y_no_noise + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(x, y, train_proportion=2/3)
    fig = go.Figure([go.Scatter(x=train_X, y=train_y, mode='lines', name='train'),
                     go.Scatter(x=test_X, y=test_y, mode='lines', name='test'),
                     go.Scatter(x=x, y=y_no_noise, mode='lines', name='true')])
    fig.update_layout(title="test and train samples noise level {}".format(noise))
    fig.write_image("./graphs/test and train samples noise level {}.png".format(noise))
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    num_of_degrees = 11
    ks = np.array(np.arange(num_of_degrees))
    error_train = np.zeros(num_of_degrees)
    error_val = np.zeros(num_of_degrees)
    for k in ks:
        pl = PolynomialFitting(k)
        error_train[k], error_val[k] = cross_validate(pl, train_X, train_y, mean_square_error)
    fig = go.Figure([go.Scatter(x=ks, y=error_train, mode='lines+markers', name='average training error'),
                     go.Scatter(x=ks, y=error_val, mode='lines+markers', name='average validation error')])
    fig.update_layout(title="average train and validation error in polynomial fitting noise level {}".format(noise))
    fig.write_image("./graphs/train validation error in polynomial fitting noise level {}.png".format(noise))


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(error_val)
    pl = PolynomialFitting(k_star)
    pl.fit(train_X, train_y)
    print('k star is {}'.format(k_star))
    print("loss is {}".format(pl.loss(test_X, test_y)))




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples, :]
    train_y = y[:n_samples]
    test_X = X[n_samples:, :]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range = np.linspace(0.001, 2, n_evaluations)
    error_train_ridge = np.zeros(n_evaluations)
    error_val_ridge = np.zeros(n_evaluations)
    error_train_lasso = np.zeros(n_evaluations)
    error_val_lasso = np.zeros(n_evaluations)
    for i in range(n_evaluations):
        ridge_reg = RidgeRegression(lambda_range[i], True)
        error_train_ridge[i], error_val_ridge[i] = cross_validate(ridge_reg, train_X, train_y, mean_square_error)
        lasso_reg = Lasso(lambda_range[i], max_iter=3500)
        error_train_lasso[i], error_val_lasso[i] = cross_validate(lasso_reg, train_X, train_y, mean_square_error)
    fig = go.Figure([go.Scatter(x=lambda_range, y=error_train_ridge, mode='lines+markers', name='average training '
                                                                                                'error ridge'),
                     go.Scatter(x=lambda_range, y=error_val_ridge, mode='lines+markers', name='average validation '
                                                                                              'error ridge'),
                     go.Scatter(x=lambda_range, y=error_train_lasso, mode='lines+markers', name='average training '
                                                                                                'error lasso'),
                     go.Scatter(x=lambda_range, y=error_val_lasso, mode='lines+markers', name='average validation '
                                                                                              'error lasso')
                     ])
    fig.update_layout(title="average train and validation error in ridge and lasso regression")
    fig.write_image("./graphs/train validation error ridge and lasso.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lambda_star_ridge = lambda_range[np.argmin(error_val_ridge)]
    ridge_reg = RidgeRegression(lambda_star_ridge, True)
    ridge_reg.fit(train_X, train_y)
    print('lambda star in ridge is {}'.format(lambda_star_ridge))
    print("loss is {}".format(ridge_reg.loss(test_X, test_y)))
    lambda_star_lasso = lambda_range[np.argmin(error_val_lasso)]
    lasso_reg = Lasso(lambda_star_lasso, max_iter=3500)
    lasso_reg.fit(train_X, train_y)
    print('lambda star in lasso is {}'.format(lambda_star_lasso))
    print("loss is {}".format(mean_square_error(test_y, lasso_reg.predict(test_X))))
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    print("loss for linear regression is {}".format(lr.loss(test_X, test_y)))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
