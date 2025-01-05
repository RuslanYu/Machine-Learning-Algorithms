# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
ml_models.py

A module that defines:
1) A flexible BaseEstimator class for future expansions (classification, clustering, PCA, etc.).
2) A concrete LinearRegressionGD class that uses Gradient Descent.

This structure allows you to extend the code to any ML task by adding new classes
that inherit from BaseEstimator (e.g., RandomForestClassifier, SVM, KMeans, PCA, NeuralNetwork, etc.).
"""

import numpy as np
from abc import ABC, abstractmethod

# =============================================================================
# 1) BaseEstimator - a flexible interface for both supervised & unsupervised ML
# =============================================================================
class BaseEstimator(ABC):
    """
    A flexible base class for all machine learning estimators.

    It provides common method signatures used by different kinds of tasks:
    - Supervised: fit(X, y), predict(X)
    - Unsupervised: fit(X), transform(X), fit_transform(X)
    - (Optionally) predict_proba, decision_function, etc. for classification.

    Subclasses can override or selectively implement the methods they need.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Train the model on data (X, y).
        For supervised tasks, y is required.
        For unsupervised tasks, y may be ignored or None.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None
        """
        pass

    def predict(self, X):
        """
        Predict target values (for regression/classification).
        Unsupervised algorithms might not implement this.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
        """
        raise NotImplementedError("predict() is not implemented for this estimator.")

    def transform(self, X):
        """
        Transform data (e.g., for PCA, clustering).
        Supervised algorithms may not implement this.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : np.ndarray
        """
        raise NotImplementedError("transform() is not implemented for this estimator.")

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it. Common in unsupervised tasks (e.g., PCA).
        """
        self.fit(X, y)
        return self.transform(X)

# =============================================================================
# 2) LinearRegressionGD - Supervised Regression with Gradient Descent
# =============================================================================
class LinearRegressionGD(BaseEstimator):
    """
    A Linear Regression model using Gradient Descent for optimization.

    Attributes
    ----------
    learning_rate : float
        Step size for parameter updates in Gradient Descent.
    epochs : int
        Number of iterations over the dataset.
    beta : np.ndarray or None
        Learned coefficients (including intercept as the first element).
    cost_history : list of float
        Stores Mean Squared Error (MSE) at each iteration.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Constructor for LinearRegressionGD.

        Parameters
        ----------
        learning_rate : float, optional
            Step size for parameter updates (default 0.01).
        epochs : int, optional
            Number of Gradient Descent iterations (default 1000).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None
        self.cost_history = []

    # ------------------------
    # Internal Helper Methods
    # ------------------------
    def _as_ndarray(self, arr, dtype=float):
        """
        Convert input to a NumPy array (if not already).
        Handles lists, tuples, Pandas objects, etc.
        """
        return np.asanyarray(arr, dtype=dtype)

    def _compute_cost(self, X, y):
        """
        Compute Mean Squared Error (MSE) for current self.beta.
        """
        y_pred = X.dot(self.beta)
        errors = y - y_pred
        mse = np.mean(errors ** 2)
        return mse

    def _compute_gradient(self, X, y):
        """
        Compute gradient of the MSE cost function w.r.t. parameters (self.beta).
        """
        n = X.shape[0]
        y_pred = X.dot(self.beta)
        errors = y - y_pred
        grad = -(2.0 / n) * X.T.dot(errors)  # gradient
        return grad

    # -----------
    # Main Methods
    # -----------
    def fit(self, X, y=None):
        """
        Fit (train) the Linear Regression model using Gradient Descent.
        For supervised learning, y must be provided.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or None

        Raises
        ------
        ValueError if y is None, because linear regression requires targets.
        """
        if y is None:
            raise ValueError("For LinearRegressionGD, 'y' cannot be None.")

        # Convert X, y to NumPy arrays
        X_ = self._as_ndarray(X)
        y_ = self._as_ndarray(y)

        # If X_ is 1D, make it 2D
        if X_.ndim == 1:
            X_ = X_.reshape(-1, 1)

        # Augment X with a column of 1s for intercept
        ones = np.ones((X_.shape[0], 1), dtype=float)
        X_aug = np.hstack((ones, X_))

        # Initialize parameters
        n_features_plus_1 = X_aug.shape[1]
        self.beta = np.zeros(n_features_plus_1, dtype=float)

        self.cost_history = []  # reset cost history

        # Gradient Descent iterations
        for _ in range(self.epochs):
            grad = self._compute_gradient(X_aug, y_)
            self.beta -= self.learning_rate * grad
            cost = self._compute_cost(X_aug, y_)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Predict target values for given X using self.beta.
        """
        X_ = self._as_ndarray(X)
        if X_.ndim == 1:
            X_ = X_.reshape(-1, 1)

        # Augment X for intercept
        ones = np.ones((X_.shape[0], 1), dtype=float)
        X_aug = np.hstack((ones, X_))

        y_pred = X_aug.dot(self.beta)
        return y_pred




     