import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from IMLearn.metrics.loss_functions import misclassification_error

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """


    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load("C:/Users/bvbor/PycharmProjects/IML.HUJI/datasets/"+f)
        X_data = data[:, :-1]
        y_data = data[:, -1]


        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def loss_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X_data, y_data))
        perp = Perceptron(callback=loss_callback)
        perp.fit(X_data, y_data)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=np.array(np.arange(len(losses))), y=losses,
                      labels={'x': 'interation num', 'y': 'loss'},
                      title="perceptron loss as a function of the iteration in the " + n + " data" )
        fig.write_image("./graphs/perceptron loss {}.png".format(n))


def get_ellipse(mu: np.ndarray, cov: np.ndarray, index=0):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load("C:/Users/bvbor/PycharmProjects/IML.HUJI/datasets/" + f)
        X_data = data[:, :-1]
        y_data = data[:, -1]

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X_data, y_data)
        y_lda = lda.predict(X_data)
        # print("lda loss in {} is {}".format(f, lda.loss(X_data, y_data)))

        gnb = GaussianNaiveBayes()
        gnb.fit(X_data, y_data)
        y_gnb = gnb.predict(X_data)
        # print("gbn loss in {} is {}".format(f, gnb.loss(X_data, y_data)))

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("GNB prediction accuracy is {}".format(accuracy(y_data, y_gnb)),
                                            "LDA prediction accuracy is {}".format(accuracy(y_data, y_lda))))

        trace = go.Scatter(x=X_data[:, 0], y=X_data[:, 1],  mode='markers',
                           marker=dict(color=y_gnb, symbol=y_data), showlegend=False)
        fig.add_trace(trace, row=1, col=1)
        trace = go.Scatter(x=X_data[:, 0], y=X_data[:, 1], mode='markers',
                           marker=dict(color=y_lda, symbol=y_data), showlegend=False)
        fig.add_trace(trace, row=1, col=2)
        fig.add_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                                 mode='markers', marker=dict(size=10, symbol=4, color="black"), showlegend=False),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                 mode='markers', marker=dict(size=10, symbol=4, color="black"), showlegend=False),
                      row=1, col=2)
        for i in range(lda.pi_.size):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]), i), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_, i), row=1, col=2)

        fig.update_layout(height=800, width=1400, title_text="Dataset: {}".format(f))
        fig.write_image("./graphs/gaussian predictions {}.png".format(f))
        # Add traces for data-points setting symbols and colors


        # Add `X` dots specifying fitted Gaussians' means


        # Add ellipses depicting the covariances of the fitted Gaussians




if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
