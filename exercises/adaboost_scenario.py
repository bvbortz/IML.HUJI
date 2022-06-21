import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    noise_str = ""
    if noise != 0:
        noise_str = " with noise"
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab = AdaBoost(DecisionStump, n_learners)
    ab.fit(train_X, train_y)
    # errors_train = np.zeros(n_learners)
    # errors_test = np.zeros(n_learners)
    # iterations = np.array(np.arange(1, n_learners+1))
    # for i in iterations:
    #     errors_train[i-1] = ab.partial_loss(train_X, train_y, i)
    #     errors_test[i-1] = ab.partial_loss(test_X, test_y, i)
    # fig = go.Figure([go.Scatter(x=iterations, y=errors_train, name="train error"),
    #                  go.Scatter(x=iterations, y=errors_test, name="test error")],
    #                 layout=go.Layout(title="AdaBoost error in train and test" + noise_str,
    #                                  xaxis=dict(title="number of weak learners"),
    #                                  yaxis=dict(title="misclassification error")))
    # fig.write_image("./graphs/AdaBoost error"+noise_str+".png")
    # # Question 2: Plotting decision surfaces
    # T = [5, 50, 100, 250]
    symbols = np.array(["circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # fig = make_subplots(rows=2, cols=2, subplot_titles=["{} iterations".format(m) for m in T])
    # for i in range(len(T)):
    #     fig.add_traces([decision_surface(lambda x: ab.partial_predict(x, T[i]), lims[0], lims[1], showscale=False),
    #                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
    #                                marker=dict(color=test_y, symbol=symbols[(0.5 * test_y + 0.5).astype(int)],
    #                                            colorscale=[custom[0], custom[-1]],
    #                                            line=dict(color="black", width=1)))],
    #                    rows=(i // 2) + 1, cols=(i % 2) + 1)
    # fig.update_layout(title="Decision surfaces Of different amount of iteration"+noise_str, height=800,
    #                   width=1000).update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.write_image("./graphs/AdaBoost Decision surfaces" + noise_str + ".png")
    # # Question 3: Decision surface of best performing ensemble
    # min_index = np.argmin(errors_test)
    # fig = go.Figure([decision_surface(lambda x: ab.partial_predict(x, min_index+1), lims[0], lims[1], showscale=False),
    #                  go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
    #                             marker=dict(color=test_y, symbol=symbols[(0.5 * test_y + 0.5).astype(int)],
    #                                         colorscale=[custom[0], custom[-1]],
    #                                         line=dict(color="black", width=1)))])
    # fig.update_layout(title="Best Decision surface "+noise_str+" in {} iteration , accuracy is {}".format(min_index+1,
    #                                                                                         1 - errors_test[min_index]))\
    #     .update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.write_image("./graphs/AdaBoost best Decision surfaces"+noise_str+".png")
    # Question 4: Decision surface with weighted samples
    D = ab.D_ / np.max(ab.D_) * 10
    fig = go.Figure(
        [decision_surface(ab.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y, line=dict(color="black", width=1),
                                colorscale=[custom[0], custom[-1]], size=D))])
    fig.update_layout(title="Decision surface with weighted samples" + noise_str) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image("./graphs/AdaBoost best Decision surfaces"+noise_str+" with distribution.png")



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
