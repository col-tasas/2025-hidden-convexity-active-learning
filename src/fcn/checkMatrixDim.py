__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "2025-09-02"

import numpy as np


def checkMatrixDim(M_list: list[np.ndarray], dir: int = 1) -> int:
    """
    Check that a list of matrices have consistent dimensions along a specified axis.

    Parameters
    ----------
    M_list : list of np.ndarray
        List of 2D numpy arrays (matrices) to check.
    dir : int, optional
        Axis to check consistency along:
        - 1: check number of rows (default)
        - 2: check number of columns

    Returns
    -------
    n : int
        Number of rows (if dir=1) or columns (if dir=2) of the matrices.

    Raises
    ------
    TypeError
        If the matrices in the list do not have consistent dimensions along the chosen axis,
        or if dir is not 1 or 2.
    """

    n = 0  # reference dimension to compare against
    for iMatrix, M in enumerate(M_list):
        if n != 0:
            dim1, dim2 = M.shape
            # For the first matrix, set the reference dimension
            if dir == 1:
                if dim1 != n:
                    raise TypeError(
                        "Dimensions of matrices in A_true need to be consistent"
                    )
            elif dir == 2:
                if dim2 != n:
                    raise TypeError(
                        "Dimensions of matrices in A_true need to be consistent"
                    )
            else:
                raise TypeError("Select dir =1 or dir = 2")
        else:
            # Check consistency of current matrix against reference
            dim1, dim2 = M.shape
            if dir == 1:
                n = dim1
            elif dir == 2:
                n = dim2
            else:
                raise TypeError("Select dir =1 or dir = 2")

    return n
