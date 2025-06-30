"""
RoCo-TRICC: Row and Columnwise Trimmed Co-Clustering

This module implements RoCo-TRICC, a robust model-based co-clustering method designed for
structured data matrices that may be contaminated with row and/or columnwise outliers.
RoCo-TRICC jointly estimates the co-clustering structure (row and column partitions),
model parameters, and detects outlying rows/columns through a trimmed Classification EM approach.

Main Components:
----------------
- `roco_tbcem`: Main entry point. Fits the robust co-clustering model using a trimmed Classification EM algorithm.
- Helper functions to:
    - Initialize partitions
    - Perform parameter and partition updates under Poisson or Gaussian models
    - Track convergence

Features:
---------
- Supports both Gaussian and (unnormalized) Poisson Latent Block Models
- Robust to rowwise and columnwise contamination
- Detects structural outliers using impartial trimming
- Different initialization strategies (random, trimmed double k-means, constrained trimmed Poisson LBM)
- Returns estimated partitions, parameters, and fitting diagnostics

Dependencies:
-------------
- Custom imports from the TCoClust package (may include third-party transitive dependencies):
    - `tcc.config`
    - `tcc.class_defs`
    - `tcc.poisson_utils`
    - `tcc.normal_utils`
    - `tcc.utils`
    - `tcc.metrics`

References:
----------
Fibbi, E., Perrotta, D., Torti, F., Van Aelst, S., Verdonck, T. "Co-clustering contaminated data: a robust model-based
approach." Adv Data Anal Classif 18, 121–161 (2024). https://doi.org/10.1007/s11634-023-00549-3

Author:
-------
Edoardo Fibbi, edoardo.fibbi(at)kuleuven.be

"""

from .config import *
from .class_defs import *
from .poisson_utils import (poisson_block_params_init, poisson_col_posterior, poisson_row_posterior, poisson_mle,
                            poisson_log_complete_likelihood)
from .normal_utils import (normal_block_params_init, normal_col_posterior, normal_row_posterior, normal_mle,
                           normal_log_complete_likelihood, block_sse)
from .utils import is_degenerate, partition_matrices_init, mixing_proportions
from .metrics import phi_squared


def roco_tbcem(X: np.ndarray,
               g: int,
               m: int,
               density: str,
               constrained: bool = False,
               equal_weights: bool = False,
               a: float = .2,
               b: float = .2,
               n_init: int = 20,
               init_strategy: Union[str, None] = None,
               init_params: str = "sample",
               criterion: str = "both",
               t_max: int = 20,
               tol: float = 1.e-16,
               until_converged: bool = False,
               seed: Union[int, None] = None) -> TccResult:
    """ roco_tbcem fits a Latent Block Model (LBM) to a data matrix X via the Trimmed Block CEM (TBCEM) algorithm.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    g : int
        Number of row groups
    m : int
        Number of column groups
    density : str
        Data distribution (currently supported options: "Poisson" for counting data, "Normal" for
        continuous data)
    constrained : bool
        If :code:`True`, in the Normal LBM block variances and mixing proportions are assumed to be equal.
        In the Poisson LBM the mixing proportions are assumed equal if :code:`True`.
        Defaults to :code:`False`.
    equal_weights : bool
        If :code:`True`. mixing proportions are assumed to be equal.
        Overwrites `constrained` if :code:`density == "Poisson"`
        Defaults to :code:`False`.
    a : float
        Row trimming level, must be between 0 and 0.5
    b : float
        Column trimming level, must be between 0 and 0.5
    n_init : int (optional)
        Number of initializations of the algorithm.
        Defaults to 20.
    init_strategy : str or None (optional)
        Determines initialization strategy, which can be:
            - random if init_strategy==None or init_strategy=="rand"
            - trimmed double k-means if init_strategy=="dkm"
            - constrained trimmed Poisson LBM if init_strategy=="poic"
        defaults to :code:`None`.
    init_params : str or np.ndarray or tuple (optional)
        Initialisation of the block parameters:
            - if init_params=="sample", the block parameters are estimated from the data
            - if a g-by-m matrix is provided, it is used as initialization
            - else, a ValueError is raised
        Defaults to "sample".
    criterion : str (optional)
        Defines the stopping criterion of the algorithm, which can be based on
        the number of iterations ("iter"), on the likelihood ("tol"), or on both.
        Defaults to "both".
    t_max : int (optional)
        Maximum number of iterations of the algorithm.
        Defaults to 20.
    tol : float (optional)
        Threshold on the absolute difference of the complete likelihood function between two consecutive iterations
        of the algorithm to enforce stopping criterion based on tolerance.
        Defaults to 1.e-16.
    until_converged : bool (optional)
        If True and no non-degenerate solution is found for the n_init initializations, then
        the algorithm is restarted until a non-degenerate solution is found.
        Defaults to :code:`False`.
    seed : None or int (optional)
        Seed of random number generator, defaults to :code:`None`. Set `seed` to a non-negative integer for
        reproducibility.

    Returns
    -------
    res : TCoClust.TccResult
        An object having the structure defined in class `TccResult`.

    """

    # todo: add controls on input

    # Input object
    Input = TccInput(X,
                     g,
                     m,
                     density,
                     constrained,
                     equal_weights,
                     a,
                     b,
                     n_init,
                     init_strategy,
                     init_params,
                     criterion,
                     t_max,
                     tol,
                     until_converged,
                     seed)

    # Start timer
    tic_ext = time.perf_counter()

    # List of results
    res_l = []

    # Random number generator
    rng = np.random.default_rng(seed=seed)

    # Constant used to avoid numerical issues in logarithms and divisions
    const = 1.e-16

    # Initialization of external iteration number and success flag
    init_num = 0
    success = False

    # External cycle to perform different initializations of TBCEM algorithm
    while _stopping_condition(init_num, success, n_init, until_converged):

        # INITIALIZATION

        # Partition matrices
        Z, W = _roco_tbcem_partitions_init(X, g, m, density, a, b, init_strategy, init_num, rng)

        # Quitting if initial partitions are degenerate
        if is_degenerate(density=density, Z=Z, W=W):
            break

        # Mixing proportions
        Pi, Rho = mixing_proportions(Z, W)

        # Block parameters
        block_params = ()
        if density == "Poisson":
            block_params = (poisson_block_params_init(init_params, X, Z, W),)
        elif density == "Normal":
            block_params = normal_block_params_init(init_params, X, Z, W, constrained)

        # TBCEM ITERATIONS

        # Initialization of iteration number, precision and lists to store log-likelihood and additional criteria
        t = 0
        eps = 1
        log_L = []
        metric = []

        # These terms can be computed only once (only used if density=="Poisson")
        log_fac_X = gammaln(X.astype(int) + 1) if density == "Poisson" else None

        # TBCEM cycle
        while _stopping_condition_inner(criterion, t_max, tol, t, eps):

            # EC-STEP (Rows)
            Z = _roco_tbcem_ec_step(0, X, density, W, a, Pi, block_params, log_fac_X, constrained,
                                    equal_weights, const)

            # M-STEP (1/2)
            Pi, Rho, block_params = _roco_tbcem_m_step(X, Z, W, density, constrained)

            # EC-STEP (Columns)
            W = _roco_tbcem_ec_step(1, X, density, Z, b, Rho, block_params, log_fac_X, constrained,
                                    equal_weights, const)

            # M-STEP (2/2)
            Pi, Rho, block_params = _roco_tbcem_m_step(X, Z, W, density, constrained)

            # Quitting if solution degenerates
            if is_degenerate(block_params, density):
                break

            # Computing complete data log-likelihood and additional criteria
            # (Phi2 for Poisson, SSE for Normal LBM)
            if density == "Poisson":
                log_L.append(poisson_log_complete_likelihood(X, Z, W, Pi, Rho, *block_params, log_fac_X,
                                                             constrained=constrained, equal_weights=equal_weights))
                metric.append(phi_squared(X, Z, W))
            elif density == "Normal":
                log_L.append(normal_log_complete_likelihood(X, Z, W, Pi, Rho, *block_params,
                                                            constrained=constrained, equal_weights=equal_weights))
                metric.append(block_sse(X, Z, W, block_params[0]))

            # Compute precision and increment iteration number
            if t > 0:
                eps = abs(log_L[t] - log_L[t - 1])
            t += 1

        # Result object
        res = TccResult(BlockParameters=block_params,
                        MixingProportions={"Pi": Pi, "Rho": Rho},
                        Partitions={"Z": Z, "W": W},
                        logL=log_L,
                        Metric=metric,
                        Success=not is_degenerate(block_params, density),
                        Input=Input)
        res_l.append(res)

        success = res.Success
        init_num += 1

    # Selecting the best result
    final_log_L = []
    for i in range(len(res_l)):
        if res_l[i].Success:
            final_log_L.append(res_l[i].logL[-1])
        else:
            final_log_L.append(-np.inf)

    res = res_l[np.argmax(final_log_L)]

    toc_ext = time.perf_counter()

    res.ElapsedTime = toc_ext - tic_ext

    return res


def _stopping_condition(init_num_, success_, n_init_, until_converged_):
    """ _stopping_condition checks if the stopping condition of the external loop of the algorithm is met

    Parameters
    ----------
    init_num_
    success_
    n_init_
    until_converged_

    Returns
    -------

    """
    if until_converged_:
        res = init_num_ < n_init_ or (init_num_ >= n_init_ and not success_)
    else:
        res = init_num_ < n_init_
    return res


def _stopping_condition_inner(criterion, t_max, tol, t, eps):
    """ _stopping_condition_inner checks if the stopping condition of the inner loop of the algorithm is met

    Parameters
    ----------
    criterion
    t_max
    tol
    t
    eps

    Returns
    -------

    """
    if criterion == "iter":
        return t < t_max
    if criterion == "tol":
        return eps > tol
    else:
        return t < t_max and eps > tol


def _roco_tbcem_ec_step(axis, X, density, P, trim_lvl, mixing_props, block_params, log_fac_X, constrained,
                        equal_weights, const):
    """ _roco_tbcem_ec_step performs the EC-step of the TBCEM algorithm

    Parameters
    ----------
    axis
    X
    density
    P
    trim_lvl
    mixing_props
    block_params
    log_fac_X
    constrained
    equal_weights
    const

    Returns
    -------

    """
    n, p = X.shape
    g, m = block_params[0].shape

    # Set parameters based on axis
    size, classes = (n, g) if axis == 0 else (p, m)
    partition_matrix = np.zeros((size, classes), dtype=int)
    posteriors = np.zeros((size, classes))

    # Compute (log) class posteriors
    if density == "Poisson":
        compute_posterior = poisson_row_posterior if axis == 0 else poisson_col_posterior
        posteriors = compute_posterior(X, P, mixing_props, *block_params, log_fac_X, constrained, equal_weights, const)
    elif density == "Normal":
        compute_posterior = normal_row_posterior if axis == 0 else normal_col_posterior
        posteriors = compute_posterior(X, P, mixing_props, *block_params, constrained, equal_weights)

    # Find the indices of max values (i.e., the class assignment)
    arg = np.argmax(posteriors, axis=1)

    # Extract the discriminant values
    d = posteriors[np.arange(size), arg]

    # Find indices to trim
    trim_count = int(np.ceil(trim_lvl * size))
    indices = np.argpartition(d, trim_count)[:trim_count]

    # Assign class labels efficiently using NumPy masking
    mask = np.ones(size, dtype=bool)
    mask[indices] = False
    partition_matrix[np.arange(size)[mask], arg[mask]] = 1

    return partition_matrix


def _roco_tbcem_m_step(X, Z, W, density, constrained):
    """ _roco_tbcem_m_step performs the M-step of the TBCEM algorithm

    Parameters
    ----------
    X
    Z
    W
    density
    constrained

    Returns
    -------

    """

    block_params = ()

    # Mixing proportions
    Pi, Rho = mixing_proportions(Z, W)

    # Block parameters
    if density == "Poisson":
        block_params = (poisson_mle(X, Z, W),)
    elif density == "Normal":
        block_params = normal_mle(X, Z, W, constrained)

    return Pi, Rho, block_params


def _roco_tbcem_partitions_init(X, g, m, density, a, b, init_strategy, init_num, rng):
    """ _roco_tbcem_partitions_init initializes the row and column partitions in the roco_tbcem method

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    g : int
        Number of row groups
    m : int
        Number of column groups
    density : str
        Data distribution (currently supported options: "Poisson" for counting data, "Normal" for
        continuous data)
    a : float
        Row trimming level, must be between 0 and 0.5
    b : float
        Column trimming level, must be between 0 and 0.5
    init_num : int
        Current iteration number of external cycle in roco_tbcem
    init_strategy : str or None
        Determines initialization strategy, which can be:
            - random if init_strategy==None or init_strategy=="rand"
            - trimmed double k-means if init_strategy=="dkm"
            - constrained trimmed Poisson LBM if init_strategy=="poic"
    rng : numpy.random.Generator

    Returns
    -------
    Z : np.ndarray
        Row partition matrix
    W : np.ndarray
        Column partition matrix

    """
    n, p = X.shape

    # Initialization via tandem trimmed k-means
    if init_strategy == "clust":
        Z, _ = roco_tbcem(X, g, 1, "Normal",
                          constrained=True, equal_weights=True,
                          a=a, b=0,
                          n_init=1, t_max=10,
                          seed=init_num).Partitions.values()
        _, W = roco_tbcem(X, 1, m, "Normal",
                          constrained=True, equal_weights=True,
                          a=0, b=b,
                          n_init=1, t_max=10,
                          seed=init_num).Partitions.values()

    # Initialization via trimmed double k-means (i.e., fully constrained Normal RoCo-TBCEM):
    elif init_strategy == "dkm":
        Z, W = roco_tbcem(X, g, m, "Normal",
                          constrained=True, equal_weights=True,
                          a=a, b=b,
                          n_init=1, t_max=5,
                          seed=init_num).Partitions.values()

    # Initialization via constrained Poisson RoCo-TBCEM:
    elif init_strategy == "poic" and density == "Poisson":
        Z, W = roco_tbcem(X, g, m, "Poisson",
                          constrained=True, equal_weights=True,
                          a=a, b=b,
                          n_init=1, t_max=5,
                          seed=init_num).Partitions.values()

    # Random initialization:
    elif init_strategy is None or init_strategy == "rand":
        Z, W = partition_matrices_init(n, p, g, m, a, b, rng=rng)

    # Wrong input resolved by random initialization (default):
    else:
        if init_strategy == "poic" and density != "Poisson":
            warnings.warn(f"\nOption init_strategy = 'poic' is only valid with a Poisson model. "
                          f"\nUsing default random initialization (init_strategy = 'rand').", stacklevel=2)
        else:
            warnings.warn(f"\n'{init_strategy}' is not a valid option for parameter init_strategy "
                          f"(possible values: 'dkm', 'poic', 'clust', 'rand', None). "
                          f"\nUsing default random initialization (init_strategy = 'rand').", stacklevel=2)
        Z, W = partition_matrices_init(n, p, g, m, a, b, rng=rng)

    return Z, W
