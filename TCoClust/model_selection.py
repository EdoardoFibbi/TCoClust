"""
Module for model selection and monitoring in (trimmed) Poisson LBMs.

This module provides functions to perform exhaustive grid search for the optimal number of row and column
clusters and trimming level (alpha) by maximizing integrated completed likelihood (ICL) criteria,
including exact and BIC-style approximations. It supports parallel processing to speed up model evaluation.

Additionally, it includes utilities for monitoring the sensitivity of estimated block parameters to
changes in the trimming level alpha, using a squared L2 distance metric (g-statistic).

Main components:
----------------
- select_model: Performs exhaustive search over grids of (g, m, alpha) parameters with optional parallelization.
- poisson_ticl: Computes the exact ICL for Poisson LBM models with trimming.
- poisson_ticl_bic: Computes a BIC-like approximation of the ICL.
- trimming_monitoring: Monitors the sensitivity (G-statistic) of block parameters to alpha.
- g_statistic: Measures squared L2 distance between block parameter estimates for different alphas.
- Helper functions for multiprocessing and model evaluation:
    - _evaluate_single_combination
    - _worker_monitoring
    - _worker_wrapper_monitoring

Features:
---------
- Exhaustive model selection over specified grids of hyperparameters
- Parallel processing support for faster model evaluation
- Currently only supports Poisson LBMs, except for `trimming_monitoring`, which can be used with any LBM

Dependencies:
-------------
- multiprocessing
- Custom imports from the TCoClust package (may include third-party transitive dependencies):
    - `tcc.config`
    - `tcc.cell_tricc`
    - `tcc.poisson_utils`

"""

import multiprocessing as mp
from .config import *
from .cell_tricc import cell_tbsem
from .poisson_utils import poisson_log_complete_likelihood


def _evaluate_single_combination(args):
    """
    Evaluate a single co-clustering model configuration and compute model selection criteria.

    Parameters
    ----------
    args : tuple
        Must contain:
            - 'g': Number of row groups
            - 'm': Number of column groups
            - 'alpha': Trimming level
            - 'data': Input data matrix
            - 'verbose': Boolean flag for verbosity
            - 'kwargs': Additional keyword arguments for the `cell_tbsem` model constructor
            - 'kwargs_icl': Keyword arguments for the ICL computation (prior hyperparameters, including beta)

    Returns
    -------
    dict
        Dictionary containing:
        - 'params': Tuple of (g, m, alpha) used for this model.
        - 'score': Tuple of (icl, icl_bic) scores.
        - 'model': The fitted `cell_tbsem` model instance.
    """

    g, m, alpha, data, verbose, kwargs, kwargs_icl = args

    try:
        model = cell_tbsem(data, g, m,
                           alpha=alpha, beta=0,
                           **kwargs)
    except Exception as e:
        if verbose:
            print(f"Failed for g={g}, m={m}, alpha={alpha}: {e}", flush=True)
        return {'params': (g, m, alpha), 'score': (-np.inf, -np.inf), 'model': None}

    # get partition matrices
    Z_hat, W_hat = model.Partitions.values()
    # get mask matrix
    _, Mo = model.M
    # get estimated parameters
    Pi_hat, Rho_hat = model.MixingProportions.values()
    Lambda_hat = model.BlockParameters

    # tICL/tICL-BIC criteria
    criteria = {
        # exact ICL
        'icl': {
            'func': poisson_ticl,
            'kwargs': kwargs_icl
        },
        # ICL-BIC approximation
        'icl_bic': {
            'func': poisson_ticl_bic,
            'kwargs': {}
        }
    }

    # tICL/tICL-BIC computation
    for criterion, cfg in criteria.items():
        func = cfg["func"]
        kwargs_criterion = cfg["kwargs"]
        if criterion == "icl_bic":
            score_iclbic = func(data, Z_hat, W_hat, Pi_hat, Rho_hat, Lambda_hat, beta=kwargs_icl["beta"], M=Mo)
        else:
            score_icl = func(data, Z_hat, W_hat, M=Mo, **kwargs_criterion)

    if verbose:
        print(f"g, m, alpha = {g}, {m}, {alpha}: done. ICL = {score_icl:.2f}, ICL-BIC = {score_iclbic:.2f}", flush=True)

    return {'params': (g, m, alpha), 'score': (score_icl, score_iclbic), 'model': model}


def select_model(data: np.ndarray,
                 row_grid: Union[List, Tuple],
                 column_grid: Union[List, Tuple],
                 alpha_grid: Union[List, Tuple],
                 kwargs: dict,
                 kwargs_icl: dict,
                 n_jobs: int = 1,
                 return_all: bool = False,
                 verbose: bool = True,
                 ) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Perform exhaustive model selection over a grid of hyperparameters for the Cell-TRICC model.

    Each model is scored using the ICL or ICL-BIC criterion. Allows for optional parallel processing.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix.
    row_grid : list or tuple
        List of candidate values for the number of row clusters (g).
    column_grid : list or tuple
        List of candidate values for the number of column clusters (m).
    alpha_grid : list or tuple
        List of candidate values for the cellwise trimming level alpha.
    kwargs : dict
        Additional keyword arguments to pass to the `cell_tbsem` model constructor.
    kwargs_icl : dict
        Keyword arguments passed to the ICL computation (prior hyperparameters, including beta).
    n_jobs : int, default=1
        Number of parallel processes to use. If 1, no multiprocessing is used.
    return_all : bool, default=False
        If True, returns all results; otherwise only the best model.
    verbose : bool, default=True
        If True, prints progress messages.

    Returns
    -------
    best_model : object
        The `cell_tbsem` model instance with the highest (exact) ICL score.
    results : list of dict
        List of results for each parameter combination (only if `return_all=True`).
        Each result contains:
            - 'params': Tuple (g, m, alpha)
            - 'score': Tuple (icl, icl_bic)
            - 'model': Trained model
    """

    # Make grid
    job_args = product(row_grid, column_grid, alpha_grid, (data,), (verbose,), (kwargs,), (kwargs_icl,))

    # Call _evaluate_single_combination for each combination (possibly in parallel)
    if n_jobs == 1:
        results = [_evaluate_single_combination(arg) for arg in job_args]
    else:
        with mp.Pool(processes=n_jobs if n_jobs > 0 else mp.cpu_count()) as pool:
            results = pool.map(_evaluate_single_combination, job_args)

    # Sort by best ICL score
    results.sort(key=lambda x: x['score'][0], reverse=True)
    best_result = results[0]

    # Print best result if verbose
    if verbose:
        print(f"Best score: {best_result['score']} with params: {best_result['params']}")

    if return_all:
        return best_result['model'], results
    return best_result['model']


def poisson_ticl(X: np.ndarray,
                 Z: np.ndarray,
                 W: np.ndarray,
                 M: Union[np.ndarray, None] = None,
                 tau: float = .5,
                 a: float = 0.01,
                 b: float = 0.01,
                 beta: float = np.log(1 / 0.001 - 1)) -> float:
    """
    poisson_ticl computes the exact integrated completed likelihood (ICL) criterion for the unnormalized Poisson
    LBM, which can be used to select the number of row and column groups, and the trimming level alpha.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    Z : np.ndarray
        Row partition matrix.
    W : np.ndarray
        Column partition matrix.
    M : np.ndarray or None
        Mask matrix if cellwise trimming is applied, otherwise set to :code:`None`.
        Defaults to :code:`None`.
    tau : float (optional)
        Parameter of the Dirichlet prior on the row and column mixing proportions.
        Defaults to 0.5, which corresponds to an uninformative Jeffrey's prior.
        We suggest using tau=4 for small or moderate data matrices.
    a : float (optional)
        Shape parameter of the Gamma prior on the block parameters.
        Defaults to 0.01.
    b : float (optional)
        Rate parameter of the Gamma prior on the block parameters.
        Defaults to 0.01.
    beta : float (optional)
        Parameter of the Boltzmann prior on the mask matrix M.
        Defaults to np.log(1 / 0.001 - 1), which corresponds to a prior expected contamination level of 0.1% and proved
        to work well in practice.

    Returns
    -------
    icl : float
        tICL value computed for the specified data matrix :math:`X` and partitions :math:`Z` and :math:`W`.

    See Also
    --------
    poisson_ticl_bic

    """

    n, g = Z.shape
    p, m = W.shape

    if M is None:
        M = np.ones((n, p), dtype=bool)
    M_all = M.copy()
    M_all[np.isnan(X)] = 0  # M_all encodes both trimming and missing cells

    if np.all(M_all):
        # case in which no cell is missing or trimmed
        Skl = Z.T @ X @ W
        Nkl = np.outer(Z.sum(0), W.sum(0))
    else:
        # at least one cell is missing or trimmed
        Skl = Z.T @ (M * np.nan_to_num(X)) @ W
        Nkl = Z.T @ M_all @ W

    icl = gammaln(tau * g) + gammaln(tau * m) - gammaln(n + tau * g) - gammaln(p + tau * m) \
          - (g + m) * gammaln(tau) - g * m * (gammaln(a) - a * np.log(b)) \
          + gammaln(Z.sum(0) + tau).sum() + gammaln(W.sum(0) + tau).sum() \
          + (gammaln(a + Skl) - (a + Skl) * np.log(b + Nkl)).sum() \
          - (M_all * gammaln(np.nan_to_num(X) + 1)).sum() \
          - beta * (1 - M).sum() - n * p * np.log(1 + np.exp(-beta))

    return icl


def poisson_ticl_bic(X: np.ndarray,
                     Z: np.ndarray,
                     W: np.ndarray,
                     Pi: np.ndarray,
                     Rho: np.ndarray,
                     Alpha: np.ndarray,
                     beta: float,
                     M: Union[np.ndarray, None] = None) -> float:
    """
    Compute the BIC-style approximation of the integrated completed likelihood (ICL) criterion
    for the cellwise trimmed unnormalized Poisson latent block model.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    Z : np.ndarray
        Row partition matrix.
    W : np.ndarray
        Column partition matrix.
    Pi : np.ndarray
        Estimated row mixing proportions.
    Rho : np.ndarray
        Estimated column mixing proportions.
    Alpha : np.ndarray
        Estimated block parameters.
    beta : float
        Boltzmann prior parameter on the mask matrix M.
    M : np.ndarray or None, optional
        Binary mask matrix indicating observed and non-trimmed entries.
        Defaults to None, in which case a full matrix is assumed.

    Returns
    -------
    icl_bic : float
        BIC-like approximation of the ICL criterion.

    See Also
    --------
    poisson_exact_icl
    """

    n, g = Z.shape
    p, m = W.shape

    log_fac_X = gammaln(np.nan_to_num(X).astype(int) + 1)
    M_not_missing = ~np.isnan(X)

    logLc = poisson_log_complete_likelihood(X, Z, W, Pi, Rho, Alpha, log_fac_X, M & M_not_missing, beta=beta)

    icl_bic = (logLc
               - n * p * np.log(1 + np.exp(-beta))
               - 0.5 * ((g - 1) * np.log(n) + (m - 1) * np.log(p) + g * m * np.log(n * p)))

    return icl_bic


def _worker_monitoring(alpha, data, g, m, verbose, kwargs):
    """
    Fit a Cell-TRICC model for a single alpha value and return block parameters.

    This function is used as a helper for evaluating the sensitivity of block parameter estimates
    to the cellwise trimming level (alpha).

    Parameters
    ----------
    alpha : float
        Trimming level to use in the model.
    data : np.ndarray
        Input data matrix.
    g : int
        Number of row clusters.
    m : int
        Number of column clusters.
    verbose : bool
        If True, prints progress messages.
    kwargs : dict
        Additional keyword arguments passed to the `cell_tbsem` constructor.

    Returns
    -------
    np.ndarray
        Estimated block parameters (Lambda_hat) from the fitted model.
    """

    res = cell_tbsem(data, g, m, alpha=alpha, **kwargs)

    if verbose:
        print(f"alpha = {alpha:.2f}: done", flush=True)

    return res.BlockParameters


def _worker_wrapper_monitoring(arg):
    """
    Wrapper for unpacking arguments when using multiprocessing with `_worker_monitoring`.

    Parameters
    ----------
    arg : tuple
        A tuple containing:
            - args : tuple
                Positional arguments for `_worker_monitoring`.
            - kwargs : dict
                Keyword arguments for `cell_tbsem`.

    Returns
    -------
    np.ndarray
        Block parameters returned by `_worker_monitoring`.
    """
    args, kwargs = arg
    return _worker_monitoring(*args, kwargs)


def trimming_monitoring(data: np.ndarray,
                        grid: Iterable,
                        g,
                        m,
                        kwargs: dict,
                        n_jobs: int = 1,
                        verbose: bool = True,
                        ) -> Iterable:
    """
    Monitors the sensitivity of the co-clustering solution with respect to the trimming parameter `alpha`.

    For each value of `alpha` in the grid, the function fits a co-clustering model and compares
    the resulting block parameters to those obtained with `alpha=0` using a squared L2 distance
    (referred to as the `g_statistic`). Optionally supports parallel execution.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix.
    grid : iterable
        Iterable of `alpha` values to evaluate (e.g., np.linspace(0, 0.1, 10)).
    g : int
        Number of row clusters.
    m : int
        Number of column clusters.
    kwargs : dict
        Additional keyword arguments passed to the `cell_tbsem` model constructor.
        Should include at least model configuration parameters; if `beta` is not provided,
        a default value of `log(1.e3 - 1)` is used.
    n_jobs : int, default=1
        Number of parallel processes to use. If 1, runs sequentially.
    verbose : bool, default=True
        If True, prints progress messages for each alpha value.

    Returns
    -------
    out : np.ndarray
        Array of shape (len(grid),) containing the g-statistic (squared L2 distance)
        between the block parameter estimates at each `alpha` and those at `alpha=0`.

    Warns
    -----
    UserWarning
        If `beta` is not present in `kwargs`, a default value is set and a warning is raised.
    """

    if "beta" not in kwargs.keys():
        warnings.warn("trimming parameter 'beta' not set in 'kwargs': using default values.")
        kwargs["beta"] = np.log(1.e3 - 1)

    job_args = [((alpha, data, g, m, verbose), kwargs) for alpha in grid]

    theta0 = cell_tbsem(data, g, m, alpha=0, **kwargs).BlockParameters

    # Call _worker_wrapper_monitoring for each combination (possibly in parallel)
    if n_jobs == 1:
        thetas1 = [_worker_wrapper_monitoring(arg) for arg in job_args]
    else:
        with mp.Pool(processes=n_jobs if n_jobs > 0 else mp.cpu_count()) as pool:
            thetas1 = pool.map(_worker_wrapper_monitoring, job_args)

    results = np.zeros(len(grid))
    for i, theta1 in enumerate(thetas1):
        results[i] = g_statistic(theta1, theta0)

    # formatting the result
    out = np.array(results)

    return out


def g_statistic(theta1, theta0):
    """
    Compute the squared L2 distance between two sorted block parameter vectors.

    This statistic is used to measure how much block parameters change with different trimming levels.

    Parameters
    ----------
    theta1 : np.ndarray
        Block parameters from model with non-zero alpha.
    theta0 : np.ndarray
        Block parameters from baseline model (alpha = 0).

    Returns
    -------
    float
        Squared L2 distance between sorted versions of theta1 and theta0.
    """
    theta1_s = np.sort(theta1.ravel())
    theta0_s = np.sort(theta0.ravel())

    return np.linalg.norm(theta1_s - theta0_s, ord=2) ** 2
