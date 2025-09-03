__author__ = ["Nicolas Chatzikiriakos", "Bowen Song"]
__contact__ = [
    "nicolas.chatzikiriakos@ist.uni-stuttgart.de",
    "bowen.song@ist.uni-stuttgart.de",
]
__date__ = "2025-09-02"


import numpy as np
import cvxpy as cp


class InputDesignAlgo:
    """
    Class to define algorithms for generating excitation signals for ARX systems.

    Supports:
        - Random excitation (rand_inputs)
        - Convex optimization-based input design (opt_inputs_cvx)
        - Wagenmaker input design (convex)
    """

    def __init__(self, n_u: int, n_y: int, p: int, q: int, gamma: float, seed=23):
        """
        Initialize the InputDesignAlgo instance.

        Parameters
        ----------
        n_u : int
            Number of input channels
        n_y : int
            Number of output channels
        p : int
            Number of past outputs in ARX model
        q : int
            Number of past inputs in ARX model
        gamma : float
            Power constraint for the input signal
        seed : int, optional
            Seed for reproducible random number generation
        """

        self.gamma = gamma
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = p * n_y + q * n_u  #  dimension of extended state
        self.q = q
        self.p = p

        # Initialize estimates of ARX matrices
        self.A_hat = [np.zeros((n_y, n_y)) for _ in range(p)]
        self.B_hat = [np.zeros((n_y, n_u)) for _ in range(q)]
        self.construct_extended_sys()

        # Covariance of extended state (placeholder)
        self.cov_x = np.zeros((self.n_x, self.n_x))

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(seed)

    # --- General Funcitons ---
    def construct_extended_sys(self):
        """
        Build the extended state-space matrices from current A_hat and B_hat estimates.

        Constructs:
            - A_extended : Extended system matrix for x_{t+1} = A_extended x_t + B_extended u_t
            - B_extended : Extended input matrix
        """

        A_11 = np.zeros((self.p * self.n_y, self.p * self.n_y))
        A_11[0 : self.n_y, :] = np.concatenate(self.A_hat, axis=1)
        A_11[self.n_y :, 0 : self.n_y * (self.p - 1)] = np.eye(self.n_y * (self.p - 1))

        v = np.zeros(self.p)
        v[0] = 1
        v = v[:, np.newaxis]  # make it a column vector (p x 1)
        A_12 = np.kron(v, np.concatenate(self.B_hat))
        A_22 = np.zeros((self.n_u * self.q, self.n_u * self.q))
        A_22[self.n_u :, 0 : self.n_u * (self.q - 1)] = np.eye(self.n_u * (self.q - 1))

        A_21 = np.zeros((self.n_u * self.q, self.n_y * self.p))

        self.A_extended = np.block([[A_11, A_12], [A_21, A_22]])
        self.B_extended = np.vstack(
            [
                np.zeros((self.p * self.n_y, self.n_u)),  # top zeros
                np.eye(self.n_u),  # middle identity
                np.zeros(((self.q - 1) * self.n_u, self.n_u)),  # bottom zeros
            ]
        )

    def identify_from_data(self, y_data: np.ndarray, x_data: np.ndarray):
        """
        Estimate the unknown matrices from collected data.

        Parameters
        ----------
        y_data : np.ndarray
            Collected output data, shape (n_y, T)
        x_data : np.ndarray
            Collected extended state data, shape (n_x, T)

        Returns
        -------
        theta_hat : np.ndarray
            Current estimate of theta_hat, shape (n_y, n_x)
        """

        cov_x = x_data @ x_data.T  # Covariance matrix of extended state

        # Use least-squares if covariance is positive definite
        if np.min(np.linalg.eigvalsh(cov_x)) > 0:
            theta_hat = y_data @ x_data.T @ np.linalg.inv(cov_x)
        else:
            theta_hat = np.zeros((self.n_y, self.n_x))

        # Update A_hat and B_hat
        A_hat_list = [
            theta_hat[:, i * self.n_y : (i + 1) * self.n_y] for i in range(self.p)
        ]
        B_hat_list = [
            theta_hat[
                :,
                self.n_y * self.p
                + i * self.n_u : self.n_y * self.p
                + (i + 1) * self.n_u,
            ]
            for i in range(self.q)
        ]

        self.A_hat = A_hat_list
        self.B_hat = B_hat_list
        self.cov_x = cov_x

        # Update extended matrices
        self.construct_extended_sys()

        return theta_hat

    # -----------------------------
    # Random input generation
    # -----------------------------
    def rand_inputs(self, T: int):
        """
        Generate a random input sequence of length T that satisfies the power constraint.

        Parameters
        ----------
        T : int
            Length of the input sequence.

        Returns
        -------
        u : np.ndarray, shape (n_u, T)
            Random input sequence satisfying the power constraint.
        """

        # Standard Gaussian scaled to meet expected energy constraint
        u = (self.gamma / np.sqrt(self.n_u)) * self.rng.standard_normal((self.n_u, T))
        return u

    # -----------------------------
    # Convex input design
    # -----------------------------
    def opt_inputs_cvx(self, k: int, T: int):
        """
         Compute the optimal input sequence in time domain from the convex problem.

         Parameters
         ----------
         k : int
             Number of frequencies.
         T : int
             Prediction horizon (time steps).

         Returns
         -------
        u_opt_t : np.ndarray
             Time-domain optimal input, shape (n_u, T)
        """

        # Solve SDP for frequency-domain matrices
        U_opt_F_list = self.convex_learning_problem_freq(k, T)

        u_opt_f = np.zeros((self.n_u, k), dtype=complex)

        # Extract principal eigenvector for each frequency
        for jj in range(k):
            U_ell = U_opt_F_list[jj]  # shape (self.n_u, self.n_u)
            D, V = np.linalg.eigh(U_ell)  # eigenvalues D, eigenvectors V
            max_index = np.argmax(D)
            # Scale eigenvector by sqrt(sum of eigenvalues)
            u_opt_f[:, jj] = np.sqrt(np.sum(np.abs(D))) * V[:, max_index]

        # Convert to time domain
        u_opt_t = self._iff_sequence(u_f=u_opt_f, T=T)

        return u_opt_t

    def convex_learning_problem_freq(self, k: int, T: int):
        """
        Formulate and solve the convex input design problem in frequency domain.

        Parameters
        ----------
        k : int
            Number of frequencies.
        T : int
            Prediction horizon.
        gamma : float
            Power constraint.

        Returns
        -------
        U_cvx.value : np.ndarray
            Optimized input matrices of shape (n_u, n_u, k), complex-valued,
            solution of the convex program.
        """

        # Define a list of complex Hermitian PSD variables for each frequency
        U_list = []
        for ell in range(k):
            U_ell = cp.Variable((self.n_u, self.n_u), complex=True, PSD=True)
            U_list.append(U_ell)

        # Define Objective
        objective = cp.Maximize(
            cp.lambda_min(self.active_learning_objective_cvx(U_list=U_list, k=k, T=T))
        )

        constraints = self.constraint_power_cvx(U_list=U_list, k=k)

        # Necessary for numerical stability of the solver (does not influence the optimal solution)
        constraints += [cp.max(cp.abs(c)) <= 10000000 for c in U_list]

        for U_ell in U_list:
            constraints += [U_ell >> 0]

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)

        prob.solve(cp.MOSEK, verbose=True)

        U_sol = [U_ell.value for U_ell in U_list]

        return U_sol

    def active_learning_objective_cvx(self, U_list: list[cp.Variable], k: int, T: int):
        """
        Compute the full covariance matrix for input design

        Parameters
        ----------
        U_list : list of cp.Variable
            List of k (n_u x n_u) Hermitian PSD variables.
        k : int
            Number of frequencies.
        T : int
            Prediction horizon.
        Returns
        -------
        cov_all : cp.Expression
            Full covariance matrix (n_x x n_x), affine in U_matrix
        """

        n_x = self.n_x
        I_nx = np.eye(n_x)

        F_list = []
        # Precompute F_ell matrices since all are constant
        for ell in range(k):
            omega = np.exp(1j * 2 * np.pi * ell / k)
            F_ell = np.linalg.inv(omega * I_nx - self.A_extended)
            F_list.append(F_ell)

        M_list = [
            cp.Constant(F_list[i] @ self.B_extended, name=f"M{i}")
            for i in range(len(F_list))
        ]

        future_cov = 0

        for ell in range(k):
            future_cov += (
                1
                / (self.gamma**2 * k**2)
                * M_list[ell]
                @ U_list[ell]
                @ M_list[ell].conj().T
            )

        cov_all = self.gamma**2 * T * future_cov + self.cov_x
        return cov_all

    def constraint_power_cvx(self, U_list: list[cp.Variable], k: int):
        """
        CVXPY-compatible power constraint: sum of traces <= gamma^2 * k^2

        Parameters
        ----------
        U_list : list of cp.Variable
            List of k (n_u x n_u) Hermitian PSD variables.
        k : int
            Number of frequencies

        Returns
        -------
        constraints : list of cp.constraints
        """

        inputEnergy = 0
        for ell in range(k):
            inputEnergy += cp.trace(U_list[ell])

        # Constraint: inputEnergy <= gamma^2 * k^2
        constraints = [inputEnergy <= self.gamma**2 * k**2]

        return constraints

    # -----------------------------
    # Wagenmaker input design (convex)
    # -----------------------------
    def opt_inputs_cvx_wagenmaker(self, k: int, T: int):
        """
        Compute the optimal input sequence in time domain from the convex problem.

        Parameters
        ----------
        k : int
            Number of frequencies.
        T : int
            Prediction horizon (time steps).

        Returns
        -------
        u : np.ndarray
            Time-domain optimal input, shape (n_u, T)
        """

        # Solve convex optimization problem for U (List of matrices)
        U_opt_F_list = self.convex_learning_problem_freq(k, T)

        u_opt_f = np.zeros((self.n_u, k, self.n_u), dtype=complex)

        # Extract vector based on Theorem 2 (maximum eigenvalue eigenvector)
        for jk, U_ell in enumerate(U_opt_F_list):
            D, V = np.linalg.eigh(U_ell)  # eigh since Hermitian
            for ii in range(self.n_u):
                u_opt_f[:, jk, ii] = np.sqrt(self.n_u * D[ii]) * V[:, ii]

        # Derive time-domain signal
        u_opt_t = np.zeros((self.n_u, T * self.n_u))
        for ii in range(self.n_u):
            u_opt_t[:, ii * T : (ii + 1) * T] = self._iff_sequence(u_opt_f[:, :, ii], T)

        return u_opt_t

    # -----------------------------
    # Fourier to time-domain helper
    # -----------------------------
    def _iff_sequence(self, u_f: np.ndarray, T: int):
        """
        Convert Fourier coefficients to a time-domain signal of length T.

        Parameters
        ----------
        u_f : np.ndarray
            Fourier coefficients, shape (n_u, k)
        T : int
            Desired length of the output time-domain signal

        Returns
        -------
        u_t : np.ndarray
            Time-domain signal, shape (n_u, T)
        """

        n_u, k = u_f.shape

        # Perform IFFT for each input channel
        u = np.zeros((n_u, k))
        for i in range(n_u):
            u[i, :] = np.real(np.fft.ifft(u_f[i, :]))

        # Repeat until length T
        repeat_factor = int(np.ceil(T / k))
        repeated_u = np.tile(u, (1, repeat_factor))
        u_t = repeated_u[:, :T]

        return u_t

    # -----------------------------
    # Helper to convert flattened vector to PSD matrices
    # -----------------------------
    def _flat2matrixList(self, U_flat, k):
        U_list = []
        for jk in range(k):
            U_jk = (
                U_flat[jk * self.n_u : (jk + 1) * self.n_u]
                + 1j
                * U_flat[
                    (self.n_u * k)
                    + jk * self.n_u : (self.n_u * k)
                    + (jk + 1) * self.n_u
                ]
            ).reshape(2, 1)
            U_matrix = U_jk @ U_jk.T
            U_list.append(U_matrix)

        return U_list
