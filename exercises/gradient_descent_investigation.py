import copy

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule, BaseLR
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weights_list = list()

    def callback_func(solver=None, weights=None, val=None, grad=None, t=None, eta=None, delta=None):
        values.append(val)
        weights_list.append(weights)
    return callback_func, values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    res_l2 = list()
    values_l2 = list()
    for eta in etas:
        lr = FixedLR(eta)
        callback_func, values, weights_list = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback_func)
        module = L2(copy.deepcopy(init))
        res_l2.append(gd.fit(module, None, None))
        values_l2.append(values)
        fig = plot_descent_path(L2, np.array(weights_list), title="fixed learning rates l2 eta={}".format(eta))
        fig.write_image("./graphs/descent_path l2 eta{} fixed.png".format(eta))
        fig = go.Figure([go.Scatter(x=np.arange(len(values)), y=values,
                                    mode='lines+markers')])
        fig.update_layout(title="fixed learning rates convergence l2 eta={}".format(eta))
        fig.write_image("./graphs/convergence l2 eta{} fixed.png".format(eta))
    res_l1 = list()
    values_l1 = list()
    for eta in etas:
        lr = FixedLR(eta)
        callback_func, values, weights_list = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback_func)
        module = L1(copy.deepcopy(init))
        res_l1.append(gd.fit(module, None, None))
        values_l1.append(values)
        fig = plot_descent_path(L1, np.array(weights_list), title="fixed learning rates l1 eta={}".format(eta))
        fig.write_image("./graphs/descent_path l1 eta{} fixed.png".format(eta))
        fig = go.Figure([go.Scatter(x=np.arange(len(values)), y=values,
                                    mode='lines+markers')])
        fig.update_layout(title="fixed learning rates convergence l1 eta={}".format(eta))
        fig.write_image("./graphs/convergence l1 eta{} fixed.png".format(eta))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    res_l1 = list()
    values_l1 = list()
    for gamma in gammas:
        lr = ExponentialLR(eta, gamma)
        callback_func, values, weights_list = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback_func)
        module = L1(copy.deepcopy(init))
        res_l1.append(gd.fit(module, None, None))
        values_l1.append(values)
        fig = plot_descent_path(L1, np.array(weights_list), title="exponential learning rates l1 gamma={}".format(gamma))
        fig.write_image("./graphs/descent_path l1 gamma{} exp.png".format(gamma))
        fig = go.Figure([go.Scatter(x=np.arange(len(values)), y=values,
                                    mode='lines+markers')])
        fig.update_layout(title="exponential learning rates convergence l1 gamma={}".format(gamma))
        fig.write_image("./graphs/convergence l1 gamma{} exp.png".format(gamma))

    # Plot algorithm's convergence for the different values of gamma

    # Plot descent path for gamma=0.95



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression()
    lr.fit(np.array(X_train), np.array(y_train))
    y_prob = lr.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(np.array(y_test), y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_image("./graphs/logistic regression roc curve.png")
    alpha_star = thresholds[np.argmax(tpr - fpr)]
    best_model = LogisticRegression(alpha=alpha_star)
    best_model.fit(np.array(X_train), np.array(y_train))
    test_loss = best_model.loss(np.array(X_test), np.array(y_test))
    print("alpha star is {} and the loss in this case is {}".format(alpha_star, test_loss))
    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation
    # to specify values
    # of regularization parameter
    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for norm in ["l1", "l2"]:
        error_train_list = list()
        error_val_list = list()
        for lamda in lamdas:
            lr = FixedLR(1e-4)
            gd = GradientDescent(max_iter=2000, learning_rate=lr)
            model = LogisticRegression(penalty=norm, solver=gd, alpha=0.5, lam=lamda)
            err_train, err_val = cross_validate(model, np.array(X_train), np.array(y_train), misclassification_error)
            error_train_list.append(err_train)
            error_val_list.append(err_val)
        lr = FixedLR(1e-4)
        gd = GradientDescent(max_iter=2000, learning_rate=lr)
        lamda_star = lamdas[np.argmin(np.array(error_val_list))]
        best_model = LogisticRegression(penalty=norm, solver=gd, alpha=0.5, lam=lamda_star)
        best_model.fit(np.array(X_train), np.array(y_train))
        test_loss = best_model.loss(np.array(X_test), np.array(y_test))
        print("for {} norm lambda star is {} and the loss in this case is {}".format(norm, lamda_star, test_loss))



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
