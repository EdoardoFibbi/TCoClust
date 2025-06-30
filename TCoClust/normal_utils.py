from .config import *
from .class_defs import *


def normal_mle(X, Z, W, constrained):
    """ Normal_MLE computes the MLEs of the Normal block densities, given the partition
        matrices Z and W. Its result is used in the M-step of TBCEM.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        constrained : bool
            If True block variances and mixing proportions are assumed to be equal
            in the model, thus only the global variance is estimated.

        Returns
        -------
        Mu : np.ndarray
            Means of Normal block densities
        Sigma : np.ndarray or float
            Variances of Normal block densities or global variance (if constrained == True)

    """

    Mu = np.linalg.multi_dot([Z.T, X, W])
    Den = np.outer(np.sum(Z, axis=0), np.sum(W, axis=0))
    Mu = np.divide(Mu,
                   Den,
                   out=np.zeros_like(Mu),
                   where=(Den != 0))

    if constrained:
        Sigma = (np.linalg.multi_dot([Z.T, X ** 2, W]) - Den * (Mu ** 2)).sum() / \
                ((Z.sum()) * (W.sum()))
    else:
        Sigma = np.linalg.multi_dot([Z.T, X ** 2, W])
        # The denominator Den stays the same
        Sigma = np.divide(Sigma,
                          Den,
                          out=np.zeros_like(Sigma),
                          where=(Den != 0))
        Sigma = Sigma - Mu ** 2

    return Mu, Sigma


def normal_block_params_init(init, X, Z, W, constrained):
    """ Normal_block_params_init initialises parameters of Normal block densities.

        Parameters
        ----------
        init : str or np.ndarray or tuple
            Initialisation of the block parameters:
                - if init=="sample", the block parameters are estimated from the data
                - if a tuple of two g-by-m matrices is provided, these are used as initialisation
                - else, a ValueError is raised
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        constrained : bool
            If True block variances and mixing proportions are assumed to be equal
            in the model

        Returns
        -------
        Mu : np.ndarray
            Means of Normal block densities
        Sigma : np.ndarray
            Variances of Normal block densities

    """

    if isinstance(init, str) and init == "sample":
        Mu, Sigma = normal_mle(X, Z, W, constrained)
    elif (isinstance(init, tuple) and len(init) == 2 and isinstance(init[0], np.ndarray) and
          isinstance(init[1], np.ndarray) and init[0].shape == init[1].shape == (Z.shape[1], W.shape[1])):
        if np.all(init[1] > 0):
            Mu = init[0]
            Sigma = init[1]
        else:
            raise ValueError("All block variances must be strictly positive.")
    else:
        raise ValueError("""Invalid initialisation of block parameters. Valid input: \n 
                         - (M, S) where M and S are g-by-m np.ndarrays and S has strictly positive elements \n
                         - "random"\n""")

    return Mu, Sigma


def normal_col_posterior(X, Z, Rho, Mu, Sigma, constrained, equal_weights):
    """ Normal_col_posterior returns discriminant functions for column classification
        in the Normal LBM, based on (log) column class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        Rho : np.ndarray
            Mixing proportions of column partition
        Mu : np.ndarray
            Means of Normal block densities
        Sigma : np.ndarray
            Variances of Normal block densities
        constrained : bool
            If True block variances and mixing proportions are assumed to be equal
            in the model, thus only the global variance is estimated.
            Overwrites "equal_weights".
        equal_weights : bool
            If :code:`True`. mixing proportions are assumed equal. If constrained == True,
            this has no effect.

        Returns
        -------
        T : np.ndarray
            Log of column posterior probabilities :math:`\mathbb{P}(W_{j,l}=1, \mathbf{X}^j | Z, \theta)`,
            up to a constant.

    """

    const = 1.e-16

    if constrained:
        T = - 0.5 * np.dot((X ** 2).T, Z).sum(axis=1).reshape((X.shape[1], 1)) + \
            np.linalg.multi_dot([X.T, Z, Mu]) - \
            0.5 * (np.dot(Z, Mu ** 2)).sum(axis=0).reshape((1, Mu.shape[1]))
    elif equal_weights:
        T = - 0.5 * (np.dot(Z, np.log(Sigma + const))).sum(axis=0) - \
            0.5 * np.linalg.multi_dot([X.T ** 2, Z, (Sigma + const) ** (-1)]) + \
            np.linalg.multi_dot([X.T, Z, Mu / (Sigma + const)]) - \
            0.5 * (np.dot(Z, Mu ** 2 / (Sigma + const))).sum(axis=0)
    else:
        T = np.log(Rho + const) - \
            0.5 * (np.dot(Z, np.log(Sigma + const))).sum(axis=0) - \
            0.5 * np.linalg.multi_dot([X.T ** 2, Z, (Sigma + const) ** (-1)]) + \
            np.linalg.multi_dot([X.T, Z, Mu / (Sigma + const)]) - \
            0.5 * (np.dot(Z, Mu ** 2 / (Sigma + const))).sum(axis=0)

    return T


def normal_row_posterior(X, W, Pi, Mu, Sigma, constrained, equal_weights):
    """ Normal_row_posterior returns discriminant functions for row classification
        in the Normal LBM, based on (log) row class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        W : np.ndarray
            Column partition matrix
        Pi : np.ndarray
            Mixing proportions of row partition
        Mu : np.ndarray
            Means of Normal block densities
        Sigma : np.ndarray
            Variances of Normal block densities
        constrained : bool
            If :code:`True`. block variances and mixing proportions are assumed to be equal
            in the model, thus only the global variance is estimated.
            Overwrites "equal_weights".
        equal_weights : bool
            If :code:`True`. mixing proportions are assumed equal. If constrained == True,
            this has no effect.

        Returns
        -------
        S : np.ndarray
            Log of column posterior probabilities :math:`\mathbb{P}(Z_{i,k}=1, \mathbf{X}_i | W, \theta)`,
            up to a constant.

    """

    const = 1.e-16

    if constrained:
        S = - 0.5 * np.dot(X ** 2, W).sum(axis=1).reshape((X.shape[0], 1)) + \
            np.linalg.multi_dot([X, W, Mu.T]) - \
            0.5 * (np.dot(W, Mu.T ** 2)).sum(axis=0).reshape((1, Mu.shape[0]))
    elif equal_weights:
        S = - 0.5 * (np.dot(W, np.log(Sigma.T + const))).sum(axis=0) - \
            0.5 * np.linalg.multi_dot([X ** 2, W, (Sigma.T + const) ** (-1)]) + \
            np.linalg.multi_dot([X, W, Mu.T / (Sigma.T + const)]) - \
            0.5 * (np.dot(W, Mu.T ** 2 / (Sigma.T + const))).sum(axis=0)
    else:
        S = np.log(Pi + const) - \
            0.5 * (np.dot(W, np.log(Sigma.T + const))).sum(axis=0) - \
            0.5 * np.linalg.multi_dot([X ** 2, W, (Sigma.T + const) ** (-1)]) + \
            np.linalg.multi_dot([X, W, Mu.T / (Sigma.T + const)]) - \
            0.5 * (np.dot(W, Mu.T ** 2 / (Sigma.T + const))).sum(axis=0)

    return S


# todo: make function to directly generate data from Normal LBM (without need to create partitions first)
# todo: rename alpha to something like 'parameters' (avoid confusion with trimming level)
def generate_normal_data(n: int,
                         p: int,
                         mu: np.ndarray,
                         sigma: np.ndarray,
                         a: float,
                         b: float,
                         row_partition0: Union[dict, list, np.ndarray],
                         col_partition0: Union[dict, list, np.ndarray],
                         seed: int) -> (np.ndarray, Union[dict, list, np.ndarray], Union[dict, list, np.ndarray]):
    """
    generate_normal_data generates a data matrix X from a Normal LBM with given parameters and partitions.

    Parameters
    ----------
    n: int
        Number of rows
    p: int
        Number of columns
    mu: np.ndarray
        Block means of Normal distributions
    sigma : np.ndarray
        Block variances of Normal distributions
    a: float
        Fraction of contaminated rows
    b: float
        Fraction of contaminated columns
    row_partition0: list or dict
        Row partition, given as list or dict of labels
    col_partition0: list or dict
        Column partition, given as list or dict of labels
    seed: int
        Seed for pseudo-random number generator

    Returns
    -------
    X: np.ndarray
        Data generated from LBM with given parameters and partitions
    row_partition: list or dict
        Row partition, given as list or dict of labels
    col_partition: list or dict
        Row partition, given as list or dict of labels

    """

    rng = np.random.default_rng(seed=seed)

    row_partition = row_partition0.copy()
    col_partition = col_partition0.copy()

    # Generating data matrix from Normal LBM
    X = np.zeros([n, p])

    for i in range(n):
        for j in range(p):
            X[i, j] = rng.normal(mu[row_partition[i], col_partition[j]],
                                 sigma[row_partition[i], col_partition[j]])

    # adding contamination
    # row outliers
    out_row = rng.choice(range(n), size=int(np.ceil(a * n)), replace=False)  # randomly select rows to contaminate
    for i in out_row:
        coin1 = rng.choice([-1, 1], size=p, replace=True)  # flip p coins
        X[i, :] = coin1 * rng.normal(1.5 * np.max(mu), 15, p)  # contaminate row i
        row_partition[i] = -1  # update partition

    # column outliers
    out_col = rng.choice(range(p), size=int(np.ceil(b * p)),
                         replace=False)  # randomly select columns to contaminate
    for j in out_col:
        coin2 = rng.choice([-1, 1], size=n, replace=True)  # flip n coins
        X[:, j] = coin2 * rng.normal(1.5 * np.max(mu), 15, n)  # contaminate column j
        col_partition[j] = -1  # update partition

    return X, row_partition, col_partition


def normal_log_complete_likelihood(X, Z, W, Pi, Rho, Mu, Sigma, constrained, equal_weights):
    """ Normal_log_L_c computes the classification (or complete-data) log-likelihood
        of the Normal LBM.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        Pi : np.ndarray
            Mixing proportions of row partition
        Rho : np.ndarray
            Mixing proportions of column partition
        Mu : np.ndarray
            Means of Normal block densities
        Sigma : np.ndarray
            Variances of Normal block densities
        constrained : bool
            If True block variances and mixing proportions are assumed to be equal
            in the model, thus only the global variance is estimated.
            Overwrites "equal_weights".
        equal_weights : bool
            If :code:`True`. mixing proportions are assumed equal. If constrained == True,
            this has no effect.

        Returns
        -------
        L : float
            Classification log-likelihood of Normal LBM for the given input.
    """

    const = 1.e-16

    if constrained:
        L = - (Z.sum() * W.sum() * np.log(Sigma)) - (1 / Sigma) * block_sse(X, Z, W, Mu)
    elif equal_weights:
        L = np.dot(np.sum(Z, axis=0), np.log(Pi)) + \
            np.dot(np.sum(W, axis=0), np.log(Rho)) + \
            - 0.5 * np.linalg.multi_dot([Z, np.log(Sigma + const), W.T]).sum() - \
            0.5 * (X ** 2 * np.linalg.multi_dot([Z, (Sigma + const) ** (-1), W.T])).sum() + \
            (X * np.linalg.multi_dot([Z, Mu / (Sigma + const), W.T])).sum() - \
            0.5 * (np.linalg.multi_dot([Z, Mu ** 2 / (Sigma + const), W.T])).sum()
    else:
        L = np.dot(np.sum(Z, axis=0), np.log(Pi)) + \
            np.dot(np.sum(W, axis=0), np.log(Rho)) - \
            0.5 * np.linalg.multi_dot([Z, np.log(Sigma), W.T]).sum() - \
            0.5 * (X ** 2 * np.linalg.multi_dot([Z, Sigma ** (-1), W.T])).sum() + \
            (X * np.linalg.multi_dot([Z, Mu / Sigma, W.T])).sum() - \
            0.5 * (np.linalg.multi_dot([Z, Mu ** 2 / Sigma, W.T])).sum()

    return L


def block_sse(X: np.ndarray,
              Z: np.ndarray,
              W: np.ndarray,
              Mu: np.ndarray) -> float:
    """ block_SSE computes the Sum of Squared Errors (SSE) of the data matrix X
        reorganised in blocks according to the partition matrices Z and W, with
        respect to the block means Mu.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix

        Returns
        -------
        sse : float
            SSE for the given input

    """

    sse = (np.linalg.multi_dot([Z.T, X ** 2, W]) - np.outer(Z.T.sum(axis=1), W.sum(axis=0)) * (Mu ** 2)).sum()

    # todo: add support for cellwise trimming

    return sse
