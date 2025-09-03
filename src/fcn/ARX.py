__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "2025-09-02"

import numpy as np


class ARX:
    """
    AutoRegressive model with eXogenous inputs (ARX).

    Represents a linear dynamical system of the form:
        y[t+1] = theta.T @ x[t] + w[t]

    Attributes
    ----------
    theta : np.ndarray
        System matrices stacked into a single parameter matrix
        of shape (n_y*p + n_u*q, n_y).
    n_y : int
        Output dimension.
    n_u : int
        Input dimension.
    p : int
        Number of output lags.
    q : int
        Number of input lags.
    """

    def __init__(self, theta: np.ndarray, n_y: int, n_u: int, p: int, q: int):
        """
        Initialize an ARX system.

        Parameters
        ----------
        theta : np.ndarray
            Stacked ARX system matrices of shape (n_y, n_y*p + n_u*q).
        n_y : int
            Output dimension.
        n_u : int
            Input dimension.
        p : int
            Number of output lags.
        q : int
            Number of input lags.
        """
        self.theta = theta.T  # store transposed for convenience
        self.n_y = n_y
        self.n_u = n_u
        self.p = p
        self.q = q

    def sim(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Simulate one step of the ARX system.

        Parameters
        ----------
        x : np.ndarray
            Extended state vector at the current timestep,
            shape (n_y*p + n_u*q,).
        w : np.ndarray
            Process noise vector, shape (n_y,).

        Returns
        -------
        y : np.ndarray
            Output of the ARX system at the next timestep,
            shape (n_y,).
        """
        return self.theta.T @ x + w
