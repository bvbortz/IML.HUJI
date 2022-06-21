from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.array(np.arange(np.min(y), np.max(y) + 1))
        k = self.classes_.size
        m = y.size
        self.pi_ = np.zeros(k)
        self.mu_ = np.zeros((k, X.shape[1]))
        self.vars_ = np.zeros((k, X.shape[1]))
        for i in range(k):
            m_k = np.count_nonzero(y == self.classes_[i])
            self.pi_[i] = m_k / m
            self.mu_[i] = 1 / m_k * (y == self.classes_[i]) @ X
            self.vars_[i] = 1 / m_k
            mu_dup = np.tile(self.mu_[i], m).reshape(X.shape)
            self.vars_[i] = 1 / m_k * (y == self.classes_[i]) @ ((X - mu_dup) ** 2)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        k = self.classes_.size
        m, d = X.shape
        from IMLearn.learners import MultivariateGaussian
        res = np.zeros((m, k))
        a = np.zeros((k, self.mu_.shape[1]))
        b = np.zeros((k, self.pi_.size))
        for i in range(k):
            mu_dup = np.tile(self.mu_[i], m).reshape(X.shape)
            res[:, i] = np.diag(self.pi_[i] / np.sqrt((2*np.pi) ** d * np.linalg.det(np.diag(self.vars_[i]))) * \
                        np.exp(-1 / 2 * (X - mu_dup) @ np.linalg.inv(np.diag(self.vars_[i])) @ np.transpose(X - mu_dup)))
        return res


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
