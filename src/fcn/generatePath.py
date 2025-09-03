__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "2025-09-02"

import os
from datetime import datetime


def generate_path_auto(base_dir="data") -> str:
    """
    Generate a unique experiment folder path with automatic numbering.

    The folder name has the format:
        expYYMMDD_<number>

    Example:
        If today is 2025-09-02 and two experiment folders already exist:
            exp250902_1
            exp250902_2
        then the next call will create and return:
            data/exp250902_3

    Parameters
    ----------
    base_dir : str, optional
        Root directory where experiment folders are stored (default is "data").

    Returns
    -------
    full_path : str
        The full path to the newly created experiment folder.
    """

    # Current date in YYMMDD format (e.g., "250902")
    date_str = datetime.now().strftime("%y%m%d")

    # Create base directory if it doesn't exist
    if not (os.path.isdir(base_dir)):
        os.mkdir(base_dir)

    # Find all folders in base_dir that match today's date pattern
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"exp{date_str}_")
    ]

    # Extract the running numbers from folder names (e.g., "exp250902_3" -> 3)
    numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]

    # Determine the next available running number (start from 1 if none exist)
    next_number = max(numbers, default=0) + 1

    # Construct the new folder name and full path
    folder_name = f"exp{date_str}_{next_number}"
    full_path = os.path.join(base_dir, folder_name)

    # Create the new directory
    os.makedirs(full_path, exist_ok=False)

    return full_path
