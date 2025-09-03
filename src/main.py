__author__ = ["Nicolas Chatzikiriakos", "Bowen Song"]
__contact__ = [
    "nicolas.chatzikiriakos@ist.uni-stuttgart.de",
    "bowen.song@ist.uni-stuttgart.de",
]
__date__ = "2025-09-02"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from fcn.checkMatrixDim import checkMatrixDim
from fcn.inputDesignAlgo import InputDesignAlgo
from fcn.constructExtendedState import constructExtendedState
from fcn.ARX import ARX
from fcn.stats import stats
from fcn.generatePath import generate_path_auto


def runExperiment(
    A_true: list[np.ndarray],
    B_true: list[np.ndarray],
    gamma: float,
    sigma_w: float,
    maxIter: int,
    propOpt: float,
    T_0: int,
    k_0: int,
    T_init: int,
    nMC: int,
    fineGrid=10,
):
    """
    Run Monte Carlo experiments for ARX system identification
    with different input design strategies.

    The function compares three approaches:
        1. Random input sequences
        2. Convex input design (ours)
        3. Convex input design (Wagenmaker)

    Parameters
    ----------
    A_true : list of np.ndarray
        List of autoregressive (AR) matrices defining the true system.
    B_true : list of np.ndarray
        List of input (X) matrices defining the true system.
    gamma : float
        Input energy bound for the design algorithms.
    sigma_w : float
        Standard deviation of the process noise.
    maxIter : int
        Number of input-design rollouts per Monte Carlo run.
    propOpt : float
        Proportion of optimal inputs versus random inputs to use (0–1).
    T_0 : int
        Initial horizon length for the input design rollout.
    k_0 : int
        Initial block length for the input design rollout.
    T_init : int
        Length of the initial data collection phase.
    nMC : int
        Number of Monte Carlo simulations.
    fineGrid: int
        Fineness for plotting (Default: 10)
    Returns
    -------
    None
        Produces plots of identification error vs. sample size.
        Saves experiment data to disk in both `.npz` (Python) and `.mat` (MATLAB) formats.
    """

    # --- System dimensions ---
    p = len(A_true)
    q = len(B_true)

    n_y = checkMatrixDim(A_true, dir=1)  # Number of outputs
    n_u = checkMatrixDim(B_true, dir=2)  # Number of inputs

    n_x = p * n_y + q * n_u  # Extended state dimension

    max_qp = np.max([p, q])  # Max lag

    rng_w = np.random.default_rng(seed=12)  # Random generator for noise

    # --- Compute maximum number of samples ---
    T_max = T_init

    for jj in range(maxIter):
        T_max = T_max + T_0 * 3 ** (jj)

    # --- True system ---
    theta_true = np.concatenate(A_true + B_true, axis=1)
    system = ARX(theta=theta_true, n_y=n_y, n_u=n_u, p=p, q=q)

    # --- Containers for estimates and results ---
    # Parameter estimates (theta_hat) for each method
    theta_hat_all = [
        np.zeros((n_y, n_x, int(np.floor(T_max / fineGrid)), nMC)),  # Random
        np.zeros((n_y, n_x, int(np.floor(T_max / fineGrid)), nMC)),  # CVX ours
        np.zeros((n_y, n_x, int(np.floor(T_max / fineGrid)), nMC)),  # CVX Wagenmaker
    ]

    # Identification errors per method
    error_all = [np.zeros((int(np.floor(T_max / fineGrid)), nMC)) for _ in range(3)]

    # Identification errors per method
    u_all = [np.zeros((n_u, T_max + max_qp, nMC)) for _ in range(3)]
    y_all = [np.zeros((n_y, T_max + max_qp, nMC)) for _ in range(3)]
    x_all = [np.zeros((n_x, T_max, nMC)) for _ in range(3)]

    # --- Initialize input design algorithms ---
    inputGen_rand = InputDesignAlgo(n_u=n_u, n_y=n_y, p=p, q=q, gamma=gamma)
    inputGen_cvx = InputDesignAlgo(n_u=n_u, n_y=n_y, p=p, q=q, gamma=gamma)
    inputGen_cvx_wagenmaker = InputDesignAlgo(n_u=n_u, n_y=n_y, p=p, q=q, gamma=gamma)

    # --- Monte Carlo simulations ---
    for jjMC in range(nMC):
        # Process noise sequence
        w_seq = sigma_w * rng_w.standard_normal((n_y, T_max))

        # Initialize inputs: random + same IC for CVX methods
        u_all[0][:, max_qp:, jjMC] = inputGen_rand.rand_inputs(T_max)
        u_all[1][:, max_qp : max_qp + T_init, jjMC] = u_all[0][
            :, max_qp : max_qp + T_init, jjMC
        ]
        u_all[2][:, max_qp : max_qp + T_init, jjMC] = u_all[0][
            :, max_qp : max_qp + T_init, jjMC
        ]

        # ===============================
        # Random input experiment
        # ===============================-
        for t in range(T_max):
            # Collect past inputs and outputs for construction of extended state
            u_seq = u_all[0][:, (t - q + max_qp) : (t + max_qp), jjMC]
            y_seq = y_all[0][:, (t - p + max_qp) : (t + max_qp), jjMC]

            # Construct extended state
            x_all[0][:, t, jjMC] = constructExtendedState(y_seq, u_seq, p, q, n_y, n_u)

            # Simulate ARX system
            y_all[0][:, t + max_qp, jjMC] = system.sim(
                x=x_all[0][:, t, jjMC], w=w_seq[:, t]
            )

            # Estimate parameters at regular intervals (for plotting)
            if t != 0 and t % fineGrid == 0:
                # Identify parameters
                theta_hat_all[0][:, :, int(t / fineGrid) - 1, jjMC] = (
                    inputGen_rand.identify_from_data(
                        y_all[0][:, max_qp : (t + max_qp + 1), jjMC],
                        x_all[0][:, : t + 1, jjMC],
                    )
                )

                # Compute error (for plotting)
                error_all[0][int(t / fineGrid) - 1, jjMC] = np.linalg.norm(
                    theta_hat_all[0][:, :, int(t / fineGrid) - 1, jjMC] - theta_true, 2
                )

        # ===============================
        # CVX-based input design (ours)
        # ===============================
        # Initialization (reuse random data)
        x_all[1][:, :T_init, jjMC] = x_all[0][:, :T_init, jjMC]
        y_all[1][:, max_qp : T_init + max_qp, jjMC] = y_all[0][
            :, max_qp : T_init + max_qp, jjMC
        ]
        theta_hat_all[1][:, :, : int(T_init / fineGrid) - 1, jjMC] = theta_hat_all[0][
            :, :, : int(T_init / fineGrid) - 1, jjMC
        ]
        error_all[1][: int(T_init / fineGrid) - 1, jjMC] = error_all[0][
            : int(T_init / fineGrid) - 1, jjMC
        ]

        # --- Rollouts for CVX (ours) ---
        T = T_init
        for jj in range(maxIter):
            T_i = T_0 * 3**jj
            k_i = k_0 * 2**jj

            # Identify parameters and design new inputs
            _ = inputGen_cvx.identify_from_data(
                y_data=y_all[1][:, max_qp : max_qp + T, jjMC],
                x_data=x_all[1][:, :T, jjMC],
            )
            u_all[1][:, T + max_qp : T + max_qp + T_i, jjMC] = (
                np.sqrt(propOpt) * inputGen_cvx.opt_inputs_cvx(k=k_i, T=T_i)
                + np.sqrt(1 - propOpt)
                * u_all[0][:, T + max_qp : T + max_qp + T_i, jjMC]
            )

            # Simulate system with new inputs
            for t in range(T, T + T_i):
                u_seq = u_all[1][:, t + max_qp - q : t + max_qp, jjMC]
                y_seq = y_all[1][:, t + max_qp - p : t + max_qp, jjMC]
                x_all[1][:, t, jjMC] = constructExtendedState(
                    y_seq=y_seq, u_seq=u_seq, p=p, q=q, n_y=n_y, n_u=n_u
                )
                y_all[1][:, t + max_qp, jjMC] = system.sim(
                    x=x_all[1][:, t, jjMC], w=w_seq[:, t]
                )

                # Estimate parameters at regular intervals (for plotting)
                if t != 0 and t % fineGrid == 0:
                    # Identify parameters
                    theta_hat_all[1][:, :, int(t / fineGrid) - 1, jjMC] = (
                        inputGen_cvx.identify_from_data(
                            y_all[1][:, max_qp : (t + max_qp + 1), jjMC],
                            x_all[1][:, : t + 1, jjMC],
                        )
                    )

                    # Compute error (for plotting)
                    error_all[1][int(t / fineGrid) - 1, jjMC] = np.linalg.norm(
                        theta_hat_all[1][:, :, int(t / fineGrid) - 1, jjMC]
                        - theta_true,
                        2,
                    )

            T += T_i

        # ===============================
        # Wagenmaker CVX input design
        # ===============================
        # Initialization (reuse random data)
        x_all[2][:, :T_init, jjMC] = x_all[0][:, :T_init, jjMC]
        y_all[2][:, max_qp : T_init + max_qp, jjMC] = y_all[0][
            :, max_qp : T_init + max_qp, jjMC
        ]
        theta_hat_all[2][:, :, : int(T_init / fineGrid) - 1, jjMC] = theta_hat_all[0][
            :, :, : int(T_init / fineGrid) - 1, jjMC
        ]
        error_all[2][: int(T_init / fineGrid) - 1, jjMC] = error_all[0][
            : int(T_init / fineGrid) - 1, jjMC
        ]

        # --- Rollouts for Wagenmaker input design ---
        jj, T = 0, T_init

        while T <= T_max:
            T_i = T_0 * 3**jj
            k_i = k_0 * 2**jj

            # Identify parameters and design new inputs
            _ = inputGen_cvx_wagenmaker.identify_from_data(
                y_data=y_all[2][:, max_qp : max_qp + T, jjMC],
                x_data=x_all[2][:, :T, jjMC],
            )
            u_opt = inputGen_cvx_wagenmaker.opt_inputs_cvx_wagenmaker(k=k_i, T=T_i)
            u_all[2][:, T + max_qp : T + max_qp + T_i * n_u, jjMC] = (
                np.sqrt(propOpt) * u_opt[:, : np.min([T_max - T, n_u * T_i])]
                + np.sqrt(1 - propOpt)
                * u_all[0][:, T + max_qp : T + max_qp + T_i * n_u, jjMC]
            )

            # Simulate system with new inputs
            for t in range(T, np.min([T_max, T + T_i * n_u])):
                u_seq = u_all[2][:, t - q + max_qp : t + max_qp, jjMC]
                y_seq = y_all[2][:, t - p + max_qp : t + max_qp, jjMC]

                x_all[2][:, t, jjMC] = constructExtendedState(
                    y_seq, u_seq, p, q, n_y, n_u
                )
                y_all[2][:, t + max_qp, jjMC] = system.sim(
                    x_all[2][:, t, jjMC], w=w_seq[:, t]
                )

                # Estimate parameters at regular intervals (for plotting)
                if t != 0 and t % fineGrid == 0:
                    # Identify parameters
                    theta_hat_all[2][:, :, int(t / fineGrid) - 1, jjMC] = (
                        inputGen_cvx_wagenmaker.identify_from_data(
                            y_all[2][:, max_qp : (t + max_qp + 1), jjMC],
                            x_all[2][:, : t + 1, jjMC],
                        )
                    )

                    # Compute error
                    error_all[2][int(t / fineGrid) - 1, jjMC] = np.linalg.norm(
                        theta_hat_all[2][:, :, int(t / fineGrid) - 1, jjMC]
                        - theta_true,
                        2,
                    )

            T += T_i * n_u
            jj += 1

    # --- Compute statistics over Monte Carlo runs ---
    mean_all, low_all, high_all = stats(error_list=error_all)

    # ===============================
    # Plot results
    # ===============================
    approaches = ["Random Inputs", "CVX", "Wagenmaker"]
    colors = plt.cm.plasma(np.linspace(0, 1, len(approaches)))

    # Plot
    plt.figure(figsize=(8, 6))
    idx = np.arange(1, T_max + 1, fineGrid)
    for jj in range(len(approaches)):
        # Random inputs
        plt.semilogy(
            idx,
            mean_all[jj] / np.linalg.norm(theta_true, 2),
            color=colors[jj],
            linewidth=2.5,
            label=approaches[jj],
        )
        plt.fill_between(
            idx,
            low_all[jj] / np.linalg.norm(theta_true, 2),
            high_all[jj] / np.linalg.norm(theta_true, 2),
            color=colors[jj],
            alpha=0.1,
        )

    # Labels, limits, legend
    plt.ylabel(
        r"$\frac{\Vert \hat{\theta}_t - \theta^*\Vert}{\Vert \theta_*\Vert}$",
        fontsize=14,
    )
    plt.xlabel("Number of Samples ", fontsize=12)
    plt.xlim([10, T_max])
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===============================
    # Save data (Python + MATLAB)
    # ===============================
    # Generate Path
    path = generate_path_auto()

    # Save for Python use
    np.savez(
        path + "/data_py",
        y_all=y_all,
        u_all=u_all,
        x_all=x_all,
        theta_hat_all=theta_hat_all,
        error_all=error_all,
        A_true=A_true,
        B_true=B_true,
        theta_true=theta_true,
        gamma=gamma,
        sigma_w=sigma_w,
        maxIter=maxIter,
        propOpt=propOpt,
        T_0=T_0,
        k_0=k_0,
        T_init=T_init,
        nMC=nMC,
        mean_all=mean_all,
        low_all=low_all,
        high_all=high_all,
        theta_norm=np.linalg.norm(theta_true, 2),
        T_max=T_max,
    )

    # Save for Matlab use
    data_dict = {
        "y_all": y_all,
        "u_all": u_all,
        "x_all": x_all,
        "theta_hat_all": theta_hat_all,
        "theta_true": theta_true,
        "error_all": error_all,
        "A_true": A_true,
        "B_true": B_true,
        "gamma": gamma,
        "sigma_w": sigma_w,
        "maxIter": maxIter,
        "propOpt": propOpt,
        "T_0": T_0,
        "k_0": k_0,
        "T_init": T_init,
        "nMC": nMC,
        "mean_all": mean_all,
        "low_all": low_all,
        "high_all": high_all,
        "theta_norm": np.linalg.norm(theta_true, 2),
        "T_max": T_max,
    }

    # Save to a .mat file
    savemat(
        path + "/data_mat.mat",
        data_dict,
    )


if __name__ == "__main__":
    # --- Example Setup ---
    A_true = [np.array([[0.7, 0.1], [0, 0.9]]), np.array([[-0.5, 0], [0.1, -0.2]])]
    B_true = [np.array([[0.05, 0], [0, 5]])]

    sigma_w = np.sqrt(1)  # Noise std
    gamma = 10  # Input energy bound

    # --- Experiment & Algorithm Paramters ---
    T_init = 50  # Initial Data (with random excitation)
    T_0 = 200
    k_0 = 10
    maxIter = 3  # Number of iterations
    propOpt = 0.5  # Allocation optimal input
    nMC = 2  # Monte carlo

    runExperiment(
        A_true=A_true,
        B_true=B_true,
        gamma=gamma,
        sigma_w=sigma_w,
        maxIter=maxIter,
        propOpt=propOpt,
        T_0=T_0,
        k_0=k_0,
        T_init=T_init,
        nMC=nMC,
    )
