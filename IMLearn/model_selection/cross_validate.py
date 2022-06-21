from __future__ import annotations

import random
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

# optional function to split with shuffle
# def cross_validate_split2(X, y, cv):
#     X_split = list()
#     y_split = list()
#     X_copy = np.copy(X)
#     y_copy = np.copy(y)
#     fold_size = y_copy.shape[0] // cv
#     for i in range(cv):
#         fold_X = list()
#         fold_y = list()
#         while fold_size > len(fold_y) and y_copy.size > 0:
#             index = np.random.randint(y_copy.shape[0])
#             fold_X.append(X_copy[index])
#             fold_y.append(y_copy[index])
#             y_copy = np.delete(y_copy, index, axis=0)
#             X_copy = np.delete(X_copy, index, axis=0)
#         X_split.append(np.array(fold_X))
#         y_split.append(np.array(fold_y))
#     return X_split, y_split

def cross_validate_split(X, y, cv):
    m = y.shape[0]
    fold_indexes = np.remainder(np.arange(m), cv)
    X_split = list()
    y_split = list()
    for i in range(cv):
        X_split.append(X[fold_indexes == i])
        y_split.append(y[fold_indexes == i])
    return X_split, y_split


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # X_split, y_split = cross_validate_split(X, y, cv)
    # train_errors = np.zeros(cv)
    # errors = np.zeros(cv)
    # for i in range(cv):
    #     X_copy = deepcopy(X_split)
    #     X_copy.pop(i)
    #     y_copy = deepcopy(y_split)
    #     y_copy.pop(i)
    #     estimator.fit(np.concatenate(X_copy), np.concatenate(y_copy))
    #     errors[i] = scoring(y_split[i], estimator.predict(X_split[i]))
    #     train_errors[i] = scoring(np.concatenate(y_copy), estimator.predict(np.concatenate(X_copy)))
    # return np.mean(train_errors), np.mean(errors)
    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)

    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv
