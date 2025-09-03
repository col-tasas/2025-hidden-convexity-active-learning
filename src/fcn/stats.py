__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "2025-09-02"

import numpy as np


def stats(error_list: list[np.ndarray], perc_low=25, perc_up=75):
    """
    Compute summary statistics (mean and percentiles) for a list of error arrays.

    Parameters
    ----------
    error_list : list of np.ndarray
        Each element is an array of shape (num_samples, num_MC) containing
        error values for one identification approach across Monte Carlo simulations.
    perc_low : float, optional
        Lower percentile to compute (default is 25).
    perc_up : float, optional
        Upper percentile to compute (default is 75).

    Returns
    -------
    mean_all : list of np.ndarray
        Mean error over Monte Carlo runs for each approach.
    low_all : list of np.ndarray
        Lower percentile of error over Monte Carlo runs for each approach.
    high_all : list of np.ndarray
        Upper percentile of error over Monte Carlo runs for each approach.
    """

    mean_all = []  # store mean of each approach
    low_all = []  # store lower percentile of each approach
    high_all = []  # store upper percentile of each approach
    for err in error_list:
        # Compute mean across Monte Carlo runs (axis=1)
        mean_all.append(np.mean(err[:, :], axis=1))

        # Compute lower percentile (perc_low) across Monte Carlo runs
        low_all.append(np.percentile(err[:, :], perc_low, axis=1))

        # Compute upper percentile (perc_low) across Monte Carlo runs
        high_all.append(np.percentile(err[:, :], perc_up, axis=1))

    return mean_all, low_all, high_all
