from .config import *
from .class_defs import *


def poisson_block_params_init(init: Union[str, np.ndarray, Tuple[np.ndarray, np.ndarray]],
                              X: np.ndarray,
                              Z: np.ndarray,
                              W: np.ndarray,
                              cell: bool = False,
                              M: Union[np.ndarray, None] = None) -> np.ndarray:
    """ poisson_block_params_init initialises parameters of Poisson block densities.

        Parameters
        ----------
        init : str or np.ndarray or tuple
            Initialisation of the block parameters:
                - if init=="sample", the block parameters are estimated from the data
                - if a g-by-m matrix is provided, it is used as initialisation
                - else, a ValueError is raised
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        cell : bool (optional)
            True if cellwise method is being used
            Defaults to :code:`False`
        M : np.ndarray or None (optional)
            Mask matrix for cellwise method
            defaults to :code:`None`

        Returns
        -------
        Lambda : np.ndarray
            Parameters of Poisson block densities

    """

    if type(init) is str and init == "sample":
        if M is None:
            Lambda = poisson_mle(X, Z, W, cell_init=cell)
        else:
            Lambda = poisson_mle(X, Z, W, cell_init=cell, M=M)
    elif type(init) is np.ndarray and init.shape == (Z.shape[1], W.shape[1]):
        if np.all(init > 0):
            Lambda = init
        else:
            raise ValueError("All block parameters must be strictly positive.")
    else:
        raise ValueError("""Invalid initialisation of block parameters. Valid input: \n 
                         - g-by-m np.ndarray of strictly positive elements \n
                         - "sample" \n""")

    return Lambda


def poisson_mle(X: np.ndarray,
                Z: np.ndarray,
                W: np.ndarray,
                cell_init: bool = False,
                M: Union[np.ndarray, None] = None) -> np.ndarray:
    """ poisson_mle computes the MLE of the Poisson block densities on the regular data, given the partition matrices Z
        and W.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        cell_init : bool (optional)
            Indicates whether performing initial estimation for cellwise algorithm (True) or not (False).
            Defaults to :code:`False`.
        M : np.ndarray (optional)
            Mask matrix, required if some cells are to be trimmed or missing.
            defaults to :code:`None`.

        Returns
        -------
        Lambda : np.ndarray
            Parameter estimates of Poisson block densities

    """

    if cell_init:
        from itertools import product

        g = Z.shape[1]
        m = W.shape[1]

        Lambda = np.zeros((g, m), dtype=float)

        if M is None:
            Xm = X.copy()
        else:
            Xm = np.ma.array(X, mask=1 - M)
        for k in range(g):
            for l in range(m):
                I = np.where(Z[:, k])[0].tolist()
                J = np.where(W[:, l])[0].tolist()

                IJ = np.array(list(product(I, J)))

                if IJ.size == 0:
                    Lambda[k, l] = 0
                else:
                    block_vals = Xm[IJ[:, 0], IJ[:, 1]].reshape(len(I), len(J))
                    Lambda[k, l] = np.ma.median(block_vals)
    else:
        if M is None:
            Lambda = np.linalg.multi_dot([Z.T, X, W])
            Den = np.outer(np.sum(Z, axis=0), np.sum(W, axis=0))
            Lambda = np.divide(Lambda,
                               Den.astype(float),
                               out=np.zeros_like(Lambda, dtype=float),
                               where=(Den != 0))
        else:
            Lambda = np.linalg.multi_dot([Z.T, np.nan_to_num(X) * M, W])
            Den = np.linalg.multi_dot([Z.T, M, W])
            Lambda = np.divide(Lambda,
                               Den.astype(float),
                               out=np.zeros_like(Lambda, dtype=float),
                               where=(Den != 0))
    return Lambda


def poisson_cell_posterior(X: np.ndarray,
                           Z: np.ndarray,
                           W: np.ndarray,
                           Lambda: np.ndarray) -> np.ndarray:
    """
    poisson_cell_posterior computes cell Poisson posterior probabilities, given the partitions
    and block parameters.

    Parameters
    ----------
    X: np.ndarray
        Data matrix
    Z: np.ndarray
        Row partition matrix
    W: np.ndarray
        Column partition matrix
    Lambda: np.ndarray
        Parameters of Poisson block densities

    Returns
    -------
    P: np.ndarray
        Poisson cell posterior probabilities (conditioned on Z and W)

    """

    g, m = Lambda.shape
    P = np.zeros(X.shape)

    for k in range(g):
        for l in range(m):
            I = np.where(Z[:, k])[0].tolist()
            J = np.where(W[:, l])[0].tolist()

            IJ = np.array(list(product(I, J)))

            Xkl = X[IJ[:, 0], IJ[:, 1]]  # .reshape(len(I), len(J))

            P[IJ[:, 0], IJ[:, 1]] = poisson.pmf(Xkl, Lambda[k, l])

    return P


def poisson_col_posterior(X: np.ndarray,
                          Z: np.ndarray,
                          Rho: np.ndarray,
                          Lambda: np.ndarray,
                          log_fac_X: np.ndarray,
                          constrained: bool,
                          equal_weights: bool,
                          const: float,
                          M: Union[np.ndarray, None] = None) -> np.ndarray:
    """ poisson_col_posterior returns discriminant functions for column classification
        in the unnormalized Poisson LBM, based on (log) column class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        Rho : np.ndarray
            Column partition weights
        Lambda : np.ndarray
            Parameters of Poisson block densities
        log_fac_X : np.ndarray
            log(X!) (arises from the denominator of the Poisson pmf)
        constrained : bool
            Mixing proportions are assumed equal if :code:`True`. Overwrites "equal_weights"
        equal_weights : bool
            Same as "constrained". Overwrites "constrained".
        const : float
            Constant to avoid log(0)
        M : np.ndarray (optional)
            Mask matrix, required if some cells are missing.
            defaults to :code:`None`.

        Returns
        -------
        T : np.ndarray
            Log of column posterior probabilities Pr(W[j,l]=1, X[:,j] | Z, Theta)
            up to a constant.

    """

    p = X.shape[1]

    if constrained or equal_weights:
        T = np.linalg.multi_dot([X.T, Z, np.log(Lambda + const)]) - \
            np.sum(np.dot(Z, Lambda), axis=0) - \
            ((np.dot(log_fac_X.T, Z)).sum(axis=1)).reshape(p, 1)
    else:
        if M is None:
            T = np.log(Rho + const) + \
                np.linalg.multi_dot([X.T, Z, np.log(Lambda + const)]) - \
                np.sum(np.dot(Z, Lambda), axis=0) - \
                ((np.dot(log_fac_X.T, Z)).sum(axis=1)).reshape(p, 1)
        else:
            T = np.log(Rho + const) + \
                np.linalg.multi_dot([(M * np.nan_to_num(X)).T, Z, np.log(Lambda + const)]) - \
                np.linalg.multi_dot([M.T, Z, Lambda]) - \
                ((np.dot((M * np.nan_to_num(log_fac_X)).T, Z)).sum(axis=1)).reshape(p, 1)

    return T


def poisson_row_posterior(X: np.ndarray,
                          W: np.ndarray,
                          Pi: np.ndarray,
                          Lambda: np.ndarray,
                          log_fac_X: np.ndarray,
                          constrained: bool,
                          equal_weights: bool,
                          const: float,
                          M: Union[np.ndarray, None] = None) -> np.ndarray:
    """ poisson_row_posterior returns discriminant functions for row classification
        in the unnormalized Poisson LBM, based on (log) row class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        W : np.ndarray
            Column partition matrix
        Pi : np.ndarray
            Row partition weights
        Lambda : np.ndarray
            Parameters of Poisson block densities
        log_fac_X : np.ndarray
            log(X!) (arises from the denominator of the Poisson pmf)
        constrained : bool
            Mixing proportions are assumed equal if :code:`True`. Overwrites "equal_weights"
        equal_weights : bool
            Same as "constrained". Overwrites "constrained".
        const : float
            Constant to avoid log(0)
        M : np.ndarray (optional)
            Mask matrix, required if some cells are missing.
            defaults to :code:`None`.

        Returns
        -------
        S : np.ndarray
            Log of row posterior probabilities Pr(Z[i,k]=1, X[i,:] | W, Theta)
            up to a constant.

    """

    n = X.shape[0]

    if constrained or equal_weights:
        S = np.linalg.multi_dot([X, W, np.log(Lambda.T + const)]) - \
            np.sum(np.dot(W, Lambda.T), axis=0) - \
            (((np.dot(log_fac_X, W)).sum(axis=1)).reshape(n, 1))
    else:
        if M is None:
            S = np.log(Pi + const) + \
                np.linalg.multi_dot([X, W, np.log(Lambda.T + const)]) - \
                np.sum(np.dot(W, Lambda.T), axis=0) - \
                (((np.dot(log_fac_X, W)).sum(axis=1)).reshape(n, 1))
        else:
            S = np.log(Pi + const) + \
                np.linalg.multi_dot([M * np.nan_to_num(X), W, np.log(Lambda.T + const)]) - \
                np.linalg.multi_dot([M, W, Lambda.T]) - \
                ((np.dot(M * np.nan_to_num(log_fac_X), W)).sum(axis=1)).reshape(n, 1)

    return S


# todo: rename alpha to something like 'parameters' (avoid confusion with trimming level)
def generate_poisson_data(n: int,
                          p: int,
                          alpha: np.ndarray,
                          a: float,
                          b: float,
                          row_partition0: Union[dict, list, np.ndarray],
                          col_partition0: Union[dict, list, np.ndarray],
                          seed: Union[int, None] = None,
                          rng: Union[None, np.random.Generator] = None) -> (
        np.ndarray, Union[dict, list, np.ndarray], Union[dict, list, np.ndarray]):
    """
    generate_poisson_data generates a data matrix X from a Poisson LBM with given parameters and partitions.

    Parameters
    ----------
    n: int
        Number of rows
    p: int
        Number of columns
    alpha: np.ndarray
        Block means of Poisson distributions
    a: float
        Fraction of contaminated rows
    b: float
        Fraction of contaminated columns
    row_partition0: list or dict
        Row partition, given as list or dict of labels
    col_partition0: list or dict
        Column partition, given as list or dict of labels
    seed: int or None (optional)
        Seed for pseudo-random number generator
        Defaults to None, which means that the random number generator will be initialized with a random seed.
    rng: np.random.Generator or None (optional)
        Random number generator to use. If None, a new Generator will be created using the seed.

    Returns
    -------
    X: np.ndarray
        Data generated from LBM with given parameters and partitions
    row_partition: list or dict
        Row partition, given as list or dict of labels
    col_partition: list or dict
        Row partition, given as list or dict of labels

    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    row_partition = row_partition0.copy()
    col_partition = col_partition0.copy()

    # generating data matrix from Poisson LBM
    X = np.zeros([n, p])

    for i in range(n):
        for j in range(p):
            X[i, j] = rng.poisson(alpha[row_partition[i], col_partition[j]])

    # adding contamination
    # row outliers
    out_row = rng.choice(range(n), size=int(np.ceil(a * n)), replace=False)  # randomly select rows to contaminate
    for i in out_row:
        coin1 = rng.choice([0, 1], size=p, replace=True)  # flip p coins
        X[i, :] = coin1 * rng.poisson(1.5 * np.max(alpha), p) + \
                  (1 - coin1) * rng.poisson(np.min(alpha), p)  # contaminate row i
        row_partition[i] = -1  # update partition

    # column outliers
    out_col = rng.choice(range(p), size=int(np.ceil(b * p)),
                         replace=False)  # randomly select columns to contaminate
    for j in out_col:
        coin2 = rng.choice([0, 1], size=n, replace=True)  # flip n coins
        X[:, j] = coin2 * rng.poisson(1.5 * np.max(alpha), n) + \
                  (1 - coin2) * rng.poisson(np.min(alpha), n)  # contaminate column j
        col_partition[j] = -1  # update partition

    return X, row_partition, col_partition


def simulate_poisson_lbm(n: int, p: int, g: int, m: int,
                         pi: Union[np.ndarray, list, tuple, None] = None,
                         rho: Union[np.ndarray, list, tuple, None] = None,
                         block_params: Union[np.ndarray, None] = None,
                         seed: Union[int, None] = None,
                         rng: Union[np.random.Generator, None] = None,
                         tau: float = 4, a: float = 1, b: float = 3) -> (np.ndarray, dict, dict):
    """
    Generate a data matrix from a Poisson Latent Block Model (LBM) with specified or randomly sampled parameters.

    This function simulates a count data matrix from a Poisson LBM. The row and column  mixing proportions (pi and rho)
    can be either specified or sampled from a symmetric Dirichlet distribution with hyperparameter tau. Similarly, the
    Poisson block parameters can be specified or sampled from a Gamma distribution with shape parameter a and scale
    parameter b.

    Parameters
    ----------
    n : int
        Number of rows in the data matrix.
    p : int
        Number of columns in the data matrix.
    g : int
        Number of latent row groups.
    m : int
        Number of latent column groups.
    pi : array-like of shape (g,), optional
        Mixing proportions for row clusters. If None, sampled from a Dirichlet distribution with parameter `tau`.
    rho : array-like of shape (m,), optional
        Mixing proportions for column clusters. If None, sampled from a Dirichlet distribution with parameter `tau`.
    block_params : ndarray of shape (g, m), optional
        Poisson rates for each block (row-group, column-group pair). If None, sampled from a Gamma(a, b) distribution.
    seed : int, optional
        Seed for random number generation. Used only if `rng` is not provided.
    rng : np.random.Generator, optional
        Numpy random generator instance. If None, one is created using the provided `seed`.
    tau : float, default=4
        Parameter for symmetric Dirichlet distributions used when sampling `pi` and `rho`.
    a : float, default=1
        Shape parameter for the Gamma distribution used to sample `block_params` if not provided.
    b : float, default=3
        Scale parameter for the Gamma distribution used to sample `block_params` if not provided.

    Returns
    -------
    X : ndarray of shape (n, p)
        Simulated Poisson data matrix.
    row_partition : dict
        Dictionary mapping row indices to their assigned cluster labels.
    col_partition : dict
        Dictionary mapping column indices to their assigned cluster labels.

    Notes
    -----
    The function ensures that each row and column group has at least one member by resampling
    partitions until no group is empty.

    See Also
    --------
    generate_poisson_data : Low-level function used for generating the matrix once partitions are fixed.
    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    if pi is None:
        pi = rng.dirichlet(tau * np.ones(g))
    if rho is None:
        rho = rng.dirichlet(tau * np.ones(m))
    if block_params is None:
        block_params = rng.gamma(shape=a, scale=b, size=(g, m))

    while True:  # to avoid generating a partition with empty row or column groups
        row_partition_sizes = rng.multinomial(n, pi)
        col_partition_sizes = rng.multinomial(p, rho)
        if np.all(row_partition_sizes > 0) and np.all(col_partition_sizes > 0):
            break

    row_labels = np.repeat(np.arange(len(pi)), row_partition_sizes)
    col_labels = np.repeat(np.arange(len(rho)), col_partition_sizes)

    rng.shuffle(row_labels)
    rng.shuffle(col_labels)

    row_partition = dict(zip(list(range(n)), row_labels))
    col_partition = dict(zip(list(range(p)), col_labels))

    X, _, _ = generate_poisson_data(n, p, block_params, 0, 0, row_partition, col_partition, seed=seed)

    return X, row_partition, col_partition


def poisson_cell_residual(X: np.ndarray,
                          Z: np.ndarray,
                          W: np.ndarray,
                          Lambda: np.ndarray,
                          kind: str = "var_stab",
                          c: float = 3 / 8) -> np.ndarray:
    """ poisson_cell_residual computes the residual of each cell once the Poisson LBM is fitted.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    Z : np.ndarray
        Row partition matrix
    W : np.ndarray
        Column partition matrix
    Lambda : np.ndarray
        Poisson block parameters
    kind : str
        Type of residuals to compute. The available options are:
            * :code:`'var_stab'`: variance-stabilizing residuals are computed, defined as
            :math:`r = 2(\sqrt{x + c} - \sqrt{\lambda + c})` where :math:`c \geq 0 `;

            * :code:`'raw'` or :code:`'linear'`: defined as :math:`r = x - \lambda`;

            * :code:`'Anscombe'`: defined as
            :math:`r_{ij} = \frac{x^{2/3} - (\lambda^{2/3} - \frac{1}{9} \lambda^{-1/3})}{\frac{2}{3} \lambda^{1/6}}`;

            * :code:`'standardized'` or :code:`'standardised'`: defined as :math:`r = \frac{x - \lambda}{\sqrt{\lambda}}`.
        Defaults to :code:`'var_stab'`.
    c : float
        Constant used in the definition of the variance-stabilizing residuals.
        Defaults to :math:`3/8`.

    Returns
    -------
    R : np.ndarray
        Cell residuals.

    Notes
    -----
    The options :code:`'Anscombe'` and :code:`'standardized'` are only suitable when all Poisson block parameters are
    strictly positive.

    """

    g, m = Lambda.shape
    R = np.zeros(X.shape)

    for k in range(g):
        for l in range(m):
            I = np.where(Z[:, k])[0].tolist()
            J = np.where(W[:, l])[0].tolist()

            IJ = np.array(list(product(I, J)))
            Xkl = X[IJ[:, 0], IJ[:, 1]].reshape((len(I), len(J)))

            if isinstance(kind, str) and kind == "var_stab":
                if c < 0:
                    raise ValueError("Parameter 'c' must be non-negative.")
                R[IJ[:, 0], IJ[:, 1]] = (2 * (np.sqrt(Xkl + c) - np.sqrt(Lambda[k, l] + c))).ravel()
            elif isinstance(kind, str) and (kind == "raw" or kind == "linear"):
                R[IJ[:, 0], IJ[:, 1]] = (Xkl - Lambda[k, l]).ravel()
            elif isinstance(kind, str) and kind == "Anscombe":
                R[IJ[:, 0], IJ[:, 1]] = (
                        (np.power(Xkl, 2 / 3) - (Lambda[k, l] ** (2 / 3) - Lambda[k, l] ** (-1 / 3) / 9)) /
                        (2 * Lambda[k, l] ** (1 / 6) / 3)).ravel()
            elif isinstance(kind, str) and (kind == "standardized" or kind == "standardised"):
                R[IJ[:, 0], IJ[:, 1]] = np.divide(Xkl - Lambda[k, l],
                                                  np.sqrt(Lambda[k, l]),
                                                  out=np.nan * np.zeros(Xkl.shape),
                                                  where=Lambda[k, l] != 0).ravel()
            else:
                raise ValueError(
                    f"{kind} is an invalid value for parameter 'kind'. \nAccepted values are: 'var_stab', 'raw' or "
                    f"'linear', 'Anscombe', 'standardized' or 'standardised'.")

    return R


def poisson_log_complete_likelihood(X: np.ndarray,
                                    Z: np.ndarray,
                                    W: np.ndarray,
                                    Pi: np.ndarray,
                                    Rho: np.ndarray,
                                    Lambda: np.ndarray,
                                    log_fac_X: np.ndarray,
                                    M: Union[np.ndarray, None] = None,
                                    beta: float = -np.log(1.e-6),
                                    constrained: bool = False,
                                    equal_weights: bool = False) -> float:
    """ poisson_log_complete_likelihood computes the classification (or complete-data) log-likelihood of the Poisson
        LBM.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        Pi : np.ndarray
            Row partition weights
        Rho : np.ndarray
            Column partition weights
        Lambda : np.ndarray
            Parameters of Poisson block densities
        log_fac_X : np.ndarray
            log(X!) (arises from the denominator of the Poisson pmf)
        M : np.ndarray (optional)
            Mask matrix. defaults to :code:`None`
        beta : float (optional)
            Parameter controlling the strength of the penalty term.
            Defaults to -log(1.e-6)
        constrained : bool (optional)
            Mixing proportions are assumed equal if :code:`True`. Overwrites "equal_weights"
            Defaults to :code:`False`
        equal_weights : bool (optional)
            Same as "constrained". Overwrites "constrained"
            Defaults to  False

        Returns
        -------
        L : float
            Classification log-likelihood of the unnormalised Poisson LBM
            for the given input.

    """

    const = 1.e-16

    if constrained or equal_weights:
        L = - np.linalg.multi_dot([Z, Lambda, W.T]).sum() + \
            (X * (np.linalg.multi_dot([Z, np.log(Lambda + const), W.T]))).sum() - \
            (np.linalg.multi_dot([Z.T, log_fac_X, W])).sum()
    else:
        if M is None:
            L = np.dot(Z.sum(axis=0), np.log(Pi)) + np.dot(W.sum(axis=0), np.log(Rho)) - \
                np.linalg.multi_dot([Z, Lambda, W.T]).sum() + \
                (X * (np.linalg.multi_dot([Z, np.log(Lambda + const), W.T]))).sum() - \
                (np.linalg.multi_dot([Z.T, log_fac_X, W])).sum()
        else:
            L = np.dot(Z.sum(axis=0), np.log(Pi)) + np.dot(W.sum(axis=0), np.log(Rho)) - \
                (M * np.linalg.multi_dot([Z, Lambda, W.T])).sum() + \
                (M * np.nan_to_num(X) * (np.linalg.multi_dot([Z, np.log(Lambda + const), W.T]))).sum() - \
                (M * np.nan_to_num(log_fac_X)).sum() - \
                beta * (1 - M).sum()

    return L
