__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "2025-09-02"

import numpy as np


def constructExtendedState(
    y_seq: np.ndarray, u_seq: np.ndarray, p: int, q: int, n_y: int, n_u: int
) -> np.ndarray:
    """
    Construct the extended state vector for an ARX model using past outputs and inputs.

    The extended state vector x is formed by stacking past outputs and inputs:
        x = [y(t), y(t-1), ..., y(t-p+1), u(t), u(t-1), ..., u(t-q+1)]

    Parameters
    ----------
    y_seq : np.ndarray
        Past outputs with shape (n_y, p), most recent last (y[:, -1] is y(t)).
    u_seq : np.ndarray
        Past inputs with shape (n_u, q), most recent last (u[:, -1] is u(t)).
    p : int
        Number of past outputs to include.
    q : int
        Number of past inputs to include.
    n_y : int
        Output dimension.
    n_u : int
        Input dimension.

    Returns
    -------
    x : np.ndarray
        Extended state vector of shape (p*n_y + q*n_u,).
    """
    # Check dimensions
    if y_seq.shape != (n_y, p):
        raise ValueError("y_seq does not meet requirements")
    if u_seq.shape != (n_u, q):
        raise ValueError("u_seq does not meet requirements")

    # Preallocate extended state
    x = np.zeros(p * n_y + q * n_u)

    # Fill in past outputs (most recent first)
    for i in range(p):
        x[i * n_y : (i + 1) * n_y] = y_seq[:, p - i - 1]

    # Fill in past inputs (most recent first)
    for i in range(q):
        x[p * n_y + i * n_u : p * n_y + (i + 1) * n_u] = u_seq[:, q - i - 1]

    return x
