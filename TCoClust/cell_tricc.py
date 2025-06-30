"""
Cell-TRICC: Cellwise Trimmed Co-Clustering

This module implements Cell-TRICC, a robust model-based co-clustering method designed for
structured data matrices that may be contaminated with cellwise outliers.
Cell-TRICC jointly estimates the co-clustering structure (row and column partitions),
model parameters, and detects outlying cells through a trimmed Stochastic EM approach.

Main Components:
----------------
- `cell_tbsem`: Main entry point. Fits the robust co-clustering model using a trimmed Stochastic EM algorithm.
- Helper functions to:
    - Perform parameter and partition updates under Poisson model
    - Perform cellwise trimming
    - Compute final estimates of partitions and parameters
    - Enforce stopping conditions for the main algorithm loop

Features:
---------
- Currently, only supports (unnormalized) Poisson Latent Block Models
- Robust to cellwise contamination, can also handle missing data
- Detects cellwise outliers using impartial trimming
- Returns estimated partitions, parameters, and fitting diagnostics

Dependencies:
-------------
- statistics
- Custom imports from the TCoClust package (may include third-party transitive dependencies):
    - `tcc.config`
    - `tcc.class_defs`
    - `tcc.poisson_utils`
    - `tcc.utils`

References:
----------
Fibbi, E., Perrotta, D., Torti, F., Van Aelst, S., Verdonck, T. "Cell-TRICC: a Model-Based Approach to Cellwise-Trimmed
Co-Clustering." [to appear]

Author:
-------
Edoardo Fibbi, edoardo.fibbi(at)kuleuven.be

"""

from statistics import mode as st_mode

from .config import *
from .class_defs import *
from .poisson_utils import poisson_block_params_init, poisson_row_posterior, poisson_col_posterior, poisson_mle, \
    poisson_log_complete_likelihood
from .utils import partition_matrices_init, mixing_proportions, is_degenerate, get_block_values


def cell_tbsem(X: np.ndarray,
               g: int,
               m: int,
               density: str,
               alpha: float = 0.1,
               beta: float = 0,
               n_init: int = 10,
               init_params: str = "sample",
               t_max: int = 20,
               t_burn_in: int = 50,
               t_refining: int = 20,
               until_converged: bool = False,
               trim_quantile: bool = False,
               trim_global: bool = True,
               impute: bool = False,
               seed: Union[int, None] = None,
               demo: bool = False) -> Union[TccResult, Tuple[TccResult, Tuple[list, list], List[Tuple[Any, Any, Any]]]]:
    """ cell_tbsem fits a specified Latent Block Model (LBM) to a given data matrix X via the Cellwise Trimmed Block SEM
        (cell-TBSEM) algorithm, implementing the cell-TRICC method.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        g : int
            Number of row groups
        m : int
            Number of column groups
        density : str
            Data distribution (currently supported option: "Poisson" for count data)
        alpha : float (optional)
            Maximal trimming level, must be in [0, 0.5)
            Defaults to 0.1
        beta : float (optional)
            Parameter controlling the strength of trimming penalisation.
            Defaults to 0 (i.e., an exact fraction alpha of cells is trimmed).
        n_init : int (optional)
            Number of initializations of the algorithm.
            Defaults to 20.
        init_params : str or np.ndarray or tuple (optional)
            Initialization of the block parameters:
                - if :code:`init_params=="sample"`, the block parameters are estimated from the data
                - if a g-by-m matrix is provided, it is used as initialization
                - else, a :code:`ValueError` is raised
            Defaults to "sample".
        t_max : int (optional)
            Number of iterations of SEM algorithm, after burn in.
            Defaults to 20.
        t_burn_in : int (optional)
            Number of burn-in iterations of SEM algorithm.
            Defaults to 50.
        t_refining : int (optional)
            Number of iterations for the Gibbs sampler in the refining step.
            Defaults to 20.
        until_converged : bool (optional)
            If :code:`True` and no non-degenerate solution is found for the n_init initializations, then
            the algorithm is restarted until a non-degenerate solution is found.
            Defaults to :code:`False`.
        trim_quantile : bool (optional)
            If :code:`True`, trimming is based on sample alpha-quantile of cell posteriors.
            Note that, if the distribution is discrete, :code:`trim_quantile=True` will not guarantee that the trimming
            constraints are enforced exactly.
            Defaults to :code:`False`.
        trim_global : bool (optional)
            If :code:`True`, the constraint on the maximal trimming level is imposed globally rather than at block
            level. If :code:`False`, the constraint is block-wise (experimental feature).
            Defaults to :code:`True`.
        impute : bool (optional)
            If :code:`True`, missing cells are imputed from their estimated block distribution. If :code:`False`,
            missing cells are trimmed upfront.
            Defaults to :code:`False`.
        seed : None or int (optional)
            Seed of random number generator. If :code:`None`, then 'unpredictable entropy will be pulled from the OS'
            (see numpy.random.default_rng documentation). Set `seed` to a positive integer for reproducibility.
            Defaults to :code:`None`.
        demo : bool (optional)
            If :code:`True`, the function returns additional objects to produce animations illustrating the algorithm.
            Defaults to :code:`False`.

        Returns
        -------
        res : TCoClust.TccResult
            An object having the structure defined in class `TccResult`.

    """

    # todo: add controls on input

    if density != "Poisson":
        raise ValueError("Currently, the only supported option for 'density' is 'Poisson'.")

    # Copying original input (when performing imputation matrix X gets modified)
    Xo = X.copy()
    # Initial mask matrix. At this stage it is convenient to mask NaNs, but later a matrix M will just flag outliers.
    M_missing = np.isnan(X)
    M_not_missing = 1 - M_missing

    # INPUT OBJECT
    Input = TccInput(Xo,
                     g,
                     m,
                     density,
                     alpha=alpha,
                     beta=beta,
                     n_init=n_init,
                     init_strategy="rand",
                     criterion="iter",
                     t_max=t_max,
                     t_burn_in=t_burn_in,
                     until_converged=until_converged,
                     seed=seed)

    tic_ext = time.perf_counter()

    res_l = []
    framesX_l = []
    framesM_l = []
    params_l = []

    n, p = X.shape
    N = np.nansum(X)
    const = 1. / (1. * N * N)  # we will use it to avoid numerical issues in logarithms and divisions

    log_fac_X = gammaln(X + 1)  # log(X!) terms, which can be pre-computed

    init_num = 0
    success = False

    rng = np.random.default_rng(seed=seed)

    # external cycle to perform different initialisations
    while _stopping_condition(init_num, success, n_init, until_converged):

        # Setting random number generator
        if seed is not None:
            rng = np.random.default_rng(seed=seed + init_num)

        # INITIALISATION

        # Partition matrices
        Z, W = partition_matrices_init(n, p, g, m, 0, 0, rng=rng)

        # Mixing proportions
        Pi, Rho = mixing_proportions(Z, W)

        # Block parameters
        if np.isnan(X).any():
            Alpha = poisson_block_params_init(init_params, X, Z, W, cell=True, M=M_not_missing)
        else:
            Alpha = poisson_block_params_init(init_params, X, Z, W, cell=True)

        # Initialise before looping over SEM iterations
        t = 0
        log_L = []
        framesX = []
        framesM = []
        params = []

        # SEM ITERATIONS
        while t < t_max + t_burn_in:

            # At the beginning of each iteration X is reinitialised with its original values (including NaNs and outl.)
            X = Xo.copy()
            # M is also (re)initialised
            M = np.ones_like(X, dtype=bool)  # 1 - np.isnan(X)

            if np.any(Z.sum(0) == 0) or np.any(W.sum(0) == 0):
                # Go to next initialisation if one or more classes are empty
                break

            # Cellwise outlier detection + imputation
            # outlier detection
            M = _cellwise_trimming(X, Z, W, M, Alpha, trim_quantile, trim_global, alpha, beta)
            # imputation step (only impute missing cells, no need to impute the flagged ones)
            if impute:
                if M_missing.any():
                    z = part_matrix_to_dict(Z)
                    w = part_matrix_to_dict(W)
                    for ij in np.argwhere(M_missing):
                        X[ij[0], ij[1]] = rng.poisson(Alpha[z[ij[0]], w[ij[1]]])
                # computing the log(X!) terms after imputation (just those corresponding to the missing values if t > 0)
                if t > 0:
                    log_fac_X[M_missing] = gammaln(X[M_missing] + 1)
            else:
                # if no imputation is performed, M encodes both missing and outlying cells
                M = M & M_not_missing

            # SE-STEP (ROWS)
            # compute (log) row class posteriors
            S = poisson_row_posterior(X, W, Pi, Alpha, log_fac_X, False, False, const, M=M)
            logDen = np.expand_dims(logsumexp(S, axis=1), 1)
            S = np.exp(S - logDen)
            Z = rng.multinomial(n=1, pvals=S)

            # M-STEP (1/2)
            # Mixing proportions
            Pi, Rho = mixing_proportions(Z, W)
            # Block parameters
            Alpha = poisson_mle(X, Z, W, M=M) if (1 - M).any() else poisson_mle(X, Z, W)
            # Check and quit if solution degenerates
            if is_degenerate(Alpha, density, Z, W, M):
                break

            # SE-STEP (COLUMNS)
            # compute (log) column class posteriors and sample W
            T = poisson_col_posterior(X, Z, Rho, Alpha, log_fac_X, False, False, const, M=M)
            logDen = np.expand_dims(logsumexp(T, axis=1), 1)
            T = np.exp(T - logDen)
            W = rng.multinomial(n=1, pvals=T)

            # M-STEP (2/2)
            # Mixing proportions
            Pi, Rho = mixing_proportions(Z, W)
            # Block parameters
            Alpha = poisson_mle(X, Z, W, M=M) if (1 - M).any() else poisson_mle(X, Z, W)
            # Check and quit if solution degenerates
            if is_degenerate(Alpha, density, Z, W, M):
                break

            # DEMO FEATURE (video)
            if demo:
                Xosort = block_sort(Xo, part_matrix_to_dict(Z), part_matrix_to_dict(W))
                framesX.append(Xosort)
                framesM.append(block_sort(M, part_matrix_to_dict(Z), part_matrix_to_dict(W)))
            params.append((Pi, Rho, Alpha))  # this is always needed in the refining step

            # Computing complete data log-likelihood
            log_L.append(
                poisson_log_complete_likelihood(Xo, Z, W, Pi, Rho, Alpha, log_fac_X, M & M_not_missing, beta, False,
                                                False))

            # increment iteration number
            t += 1

        # Refining step (get final estimator and partitions)
        if len(params) == t_burn_in + t_max:
            # get final parameter estimates
            Pi, Rho, Alpha = _final_sem_mle(params, t_burn_in)
            # get final mask matrix
            M = np.ones_like(X, dtype=bool) if impute else ~np.isnan(X)  # reinitialize
            M = _cellwise_trimming(X, Z, W, M, Alpha, trim_quantile, trim_global, alpha, beta)
            # get final partitions
            Z, W, X, M = _sem_refining_step(X, Xo, M, W, Pi, Rho, Alpha, log_fac_X, alpha=alpha, beta=beta,
                                            n_sample=t_refining, trim_quantile=trim_quantile, trim_global=trim_global,
                                            impute=impute, rng=rng)
            log_fac_X = gammaln(X + 1)

        # Adding to Input object cleaned data matrix (with imputed cells for both NaNs and outliers)
        Input.CleanedX = X if impute else None

        # final likelihood value
        if not is_degenerate(Alpha, density, Z, W, M):
            log_L.append(poisson_log_complete_likelihood(Xo, Z, W, Pi, Rho, Alpha, log_fac_X, M & M_not_missing, beta,
                                                         False, False))

        res = TccResult(BlockParameters=Alpha,
                        MixingProportions={'Pi': Pi, 'Rho': Rho},
                        Partitions={'Z': Z, 'W': W},
                        M=(M_not_missing, M),
                        logL=log_L,
                        Success=not is_degenerate(Alpha, density, Z, W, M),
                        Input=Input)

        res_l.append(res)
        framesX_l.append(framesX)
        framesM_l.append(framesM)
        params_l.append(params)

        success = res.Success
        init_num += 1

    # SELECT FROM res_l TOP RESULT AND ASSIGN IT TO res. RETURN res
    final_log_L = []
    for i in range(len(res_l)):
        if res_l[i].Success:
            final_log_L.append(res_l[i].logL[-1])
        else:
            final_log_L.append(-np.inf)

    idx_max = np.nanargmax(final_log_L)
    res = res_l[idx_max]
    framesX = framesX_l[idx_max]
    framesM = framesM_l[idx_max]
    params = params_l[idx_max]

    toc_ext = time.perf_counter()

    res.ElapsedTime = toc_ext - tic_ext
    res.Cellwise = True

    if demo:
        return res, (framesX, framesM), params
    return res


def _stopping_condition(init_num_, success_, n_init_, until_converged_):
    if until_converged_:
        res = init_num_ < n_init_ or (init_num_ >= n_init_ and not success_)
    else:
        res = init_num_ < n_init_
    return res


def _cellwise_trimming(X, Z, W, M, Alpha, trim_quantile, trim_global, alpha, beta):
    g, m = Z.shape[1], W.shape[1]

    if not trim_global:
        for k in range(g):
            for l in range(m):

                block_vals, IJ = get_block_values(X, Z, W, k, l)
                P = poisson.pmf(block_vals, Alpha[k, l])

                if not np.isnan(P).all():
                    if trim_quantile:
                        q = np.nanquantile(P, alpha)
                        IJo = np.where((P < np.exp(-beta)) & (P < q))
                    else:
                        IJo = np.where((P < np.exp(-beta)) & _cells_to_trim(P, alpha))
                    IJok = np.fliplr(np.rot90(np.array(IJo), k=-1))
                    for i in range(IJok.shape[0]):
                        IJM = IJ.reshape((*block_vals.shape, 2))[IJok[i, 0], IJok[i, 1]]
                        M[IJM[0], IJM[1]] = 0
    else:
        P = np.zeros_like(X, dtype=float)
        for k in range(g):
            for l in range(m):
                Xkl, IJ = get_block_values(X, Z, W, k, l)
                if Xkl.size == 0:  # Skip if no elements in the block
                    continue
                Pkl = poisson.pmf(Xkl, mu=Alpha[k, l])
                P[IJ[:, 0], IJ[:, 1]] = Pkl.ravel()
        if trim_quantile:
            q = np.nanquantile(P, alpha)
            IJo = np.where((P < np.exp(-beta)) & (P < q))
        else:
            IJo = np.where((P < np.exp(-beta)) & _cells_to_trim(P, alpha))
        M[IJo] = 0
    return M


def _final_sem_mle(theta: list,
                   t_burn_in: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    _final_sem_mle computes the final SEM estimator of model parameters as the mean of their estimations
    produced after the burn-in period.

    Parameters
    ----------
    theta : list
        List of model parameters estimates. Every element of the list is expected to be a tuple containing the estimates
        at a given iteration of the SEM-Gibbs algorithm, i.e., a tuple of the form (pi, rho, alpha).
    t_burn_in: int
        Number of burn-in iterations of the SEM-Gibbs algorithm.

    Returns
    -------
    A tuple of the form (pi, rho, alpha) containing the final estimates of the model parameters.

    """

    t_tot = len(theta)
    pi_l = []
    rho_l = []
    alpha_l = []

    for i in range(t_burn_in, t_tot):
        pi_l.append(theta[i][0])
        rho_l.append(theta[i][1])
        alpha_l.append(theta[i][2].ravel())

    for i in range(t_tot - t_burn_in - 1, 0, -1):
        pi_l[i - 1] = pi_l[i - 1][pi_l[i - 1].argsort()][pi_l[i].argsort().argsort()]
        rho_l[i - 1] = rho_l[i - 1][rho_l[i - 1].argsort()][rho_l[i].argsort().argsort()]
        alpha_l[i - 1] = alpha_l[i - 1][alpha_l[i - 1].argsort()][alpha_l[i].argsort().argsort()]

    pi_final = np.zeros(pi_l[0].shape)
    rho_final = np.zeros(rho_l[0].shape)
    alpha_final = np.zeros(alpha_l[0].shape)
    for i in range(1, t_tot - t_burn_in):
        pi_final += pi_l[i]
        rho_final += rho_l[i]
        alpha_final += alpha_l[i]

    pi_final = pi_final / (t_tot - t_burn_in - 1)
    rho_final = rho_final / (t_tot - t_burn_in - 1)
    alpha_final = alpha_final / (t_tot - t_burn_in - 1)

    return pi_final, rho_final, alpha_final.reshape(theta[0][2].shape)


def _sem_refining_step(X: np.ndarray,
                       Xo: np.ndarray,
                       M: np.ndarray,
                       W: np.ndarray,
                       Pi: np.ndarray,
                       Rho: np.ndarray,
                       Alpha: np.ndarray,
                       log_fac_X: np.ndarray,
                       constrained: bool = False,
                       equal_weights: bool = False,
                       alpha: float = 0.1,
                       beta: float = 0,
                       const: float = 1.e-16,
                       n_sample: int = 10,
                       trim_quantile: bool = False,
                       trim_global: bool = False,
                       impute: bool = True,
                       seed: int = 0,
                       rng: Union[None, np.random.Generator] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    _sem_refining_step computes the final estimated partitions Z, W and imputed data X for the SEM-Gibbs
    algorithm, by computing the mode of their empirical distribution once the final parameter estimates are computed.

    Parameters
    ----------
    X : np.ndarray
        Cleaned data matrix.
    Xo : np.ndarray
        Original data matrix.
    M : np.ndarray
        Mask matrix.
    W : np.ndarray
        Column partition matrix.
    Pi : np.ndarray
        Final estimate of row mixing proportions.
    Rho : np.ndarray
        Final estimate of column mixing proportions.
    Alpha : np.ndarray
        Final estimate of block parameters.
    log_fac_X : np.ndarray
        :math:`\log(X!)`, where :math:`X` is the cleaned data matrix.
    constrained : bool (optional)
        If :code:`True`, in the Normal LBM block variances and mixing proportions are assumed to be equal.
        In the Poisson LBM the mixing proportions are assumed equal if :code:`True`.
        Defaults to :code:`False`.
    equal_weights : bool (optional)
        If :code:`True`, mixing proportions are assumed to be equal.
        Overwrites "constrained" if :code:`density == "Poisson"`.
        Defaults to :code:`False`.
    alpha : float (optional)
        Trimming level, must be in :math:`[0, 0.5)`.
        Defaults to 0.1.
    beta : float (optional)
        Parameter controlling the strength of the cellwise trimming penalty term.
        Defaults to 0.
    const : float (optional)
        Constant used to avoid numerical issues.
        Defaults to 1.e-16.
    n_sample : int (optional)
        Number of samples to estimate the mode of :math:`Z`, :math:`W` and :math:`X`.
        Defaults to 10.
    trim_quantile : bool (optional)
        If :code:`True`, trimming is based on sample alpha-quantile of cell posteriors.
        Note that, if the distribution is discrete, :code:`trim_quantile=True` will not guarantee that the trimming
        constraints are enforced exactly.
        Defaults to :code:`False`.
    trim_global : bool (optional)
        If :code:`True`, the constraint on the maximal trimming level is imposed globally rather than at block level.
        Defaults to :code:`False`.
    impute : bool (optional)
        If :code:`True`, missing cells are imputed from their estimated block distribution. If :code:`False`,
        missing cells are trimmed upfront.
        Defaults to :code:`True`.
    seed : int (optional)
        Seed to ensure reproducibility, used only if `rng` is :code:`None`.
        Defaults to 0.
    rng : numpy.random.Generator (optional)
        Random number generator object to ensure reproducibility.
        This is the recommended way of ensuring reproducibility, since it avoids accidentally using the same seed
        from other parts of the code.
        If :code:`None`, then `seed` is used to initialise a new random number generator.
        Defaults to None.

    Returns
    -------
    Z : np.ndarray
        Final estimate of the row partition matrix, obtained from the mode of its posterior distribution.
    W : np.ndarray
        Final estimate of the column partition matrix, obtained from the mode of its posterior distribution.
    X : np.ndarray
        Final estimate of the cleaned (imputed) data matrix, obtained from the mode of its posterior distribution.
    M : np.ndarray
        Final estimate of the mask matrix, obtained via the outlier detection step using the final estimates of
        :math:`Z` and :math:`W`.


    Notes
    -----
    Optional parameters `constrained` and `equal_weights` are not used by the cellwise method and are included only for
    code compatibility reasons.

    """

    n, p = X.shape
    g, m = Pi.size, Rho.size

    # initialise
    Z_sum = np.zeros([n, g])
    W_sum = np.zeros([p, m])
    X_imp = np.zeros([n_sample, n, p])

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # sample from posterior distributions (Gibbs sampling)
    for ii in range(n_sample):

        # SE-STEP (ROWS)
        S = poisson_row_posterior(X, W, Pi, Alpha, log_fac_X, constrained, equal_weights, const, M=M)
        logDen = np.expand_dims(logsumexp(S, axis=1), 1)
        S = np.exp(S - logDen)
        Z = rng.multinomial(n=1, pvals=S)
        # update
        Z_sum += Z

        # SE-STEP (COLUMNS)
        T = poisson_col_posterior(X, Z, Rho, Alpha, log_fac_X, constrained, equal_weights, const, M=M)
        logDen = np.expand_dims(logsumexp(T, axis=1), 1)
        T = np.exp(T - logDen)
        W = rng.multinomial(n=1, pvals=T)
        # update
        W_sum += W

        # OUTLIER DETECTION
        if not trim_global:
            M = 1 - np.isnan(Xo)
            for k in range(g):
                for l in range(m):
                    if not is_degenerate(Alpha, "Poisson", Z, W):
                        block_vals, IJ = get_block_values(Xo, Z, W, k, l)
                        P = poisson.pmf(block_vals, Alpha[k, l])

                        if not np.isnan(P).all():
                            if trim_quantile:
                                q = np.nanquantile(P, alpha)
                                IJo = np.where((P < np.exp(-beta)) & (P < q))  # (P <= q)
                            else:
                                IJo = np.where((P < np.exp(-beta)) & _cells_to_trim(P, alpha))
                            IJok = np.fliplr(np.rot90(np.array(IJo), k=-1))
                            for i in range(IJok.shape[0]):
                                IJM = IJ.reshape((*block_vals.shape, 2))[IJok[i, 0], IJok[i, 1]]
                                M[IJM[0], IJM[1]] = 0
        else:
            pass

        # IMPUTATION
        X = Xo.copy()
        if impute:
            if np.isnan(X).any() or (1 - M).any():
                z = part_matrix_to_dict(Z)
                w = part_matrix_to_dict(W)
                for ij in np.argwhere(np.isnan(X) | (M == 0)):
                    X[ij[0], ij[1]] = rng.poisson(Alpha[z[ij[0]], w[ij[1]]])
                    log_fac_X[ij[0], ij[1]] = gammaln(X[ij[0], ij[1]] + 1)
                X_imp[ii, :, :] = X
        else:
            pass

    # compute the mode as the final estimator of Z
    idxZ = Z_sum.argmax(axis=1)
    Z = np.zeros([n, g])
    Z[range(n), idxZ] = 1

    # compute the mode as the final estimator of W
    idxW = W_sum.argmax(axis=1)
    W = np.zeros([p, m])
    W[range(p), idxW] = 1

    if impute:
        # compute the mode as the final estimator of the imputed data cells
        if np.isnan(Xo).any() or (1 - M).any():
            X_mode = _fast_tensor_mode(X_imp)
            for ij in np.argwhere(np.isnan(Xo) | (M == 0)):
                X[ij[0], ij[1]] = X_mode[ij[0], ij[1]]
    else:
        pass

    return Z, W, X, M


def _fast_tensor_mode(X):
    _, n, p = X.shape
    out = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            out[i, j] = st_mode(X[:, i, j])
    return out


def _cells_to_trim(P: np.ndarray,
                   alpha: float) -> np.ndarray:
    """
        Parameters
        ----------
        P : np.ndarray
            Data matrix
        alpha : float
            Trimming level, must be in [0, 0.5)

        Returns
        -------
         to_be_trimmed: np.ndarray
            Matrix of bools with True cells corresponding to cells in the lower alpha-proportion of P
    """

    to_be_trimmed = np.zeros(P.shape, dtype=bool)

    if alpha < 1:
        n_trim = int(alpha * P.size)
        idx = np.argpartition(P.ravel(), n_trim)[:n_trim]
        idx = np.column_stack(np.unravel_index(idx, P.shape))
        to_be_trimmed[idx[:, 0], idx[:, 1]] = True
    else:
        to_be_trimmed = ~to_be_trimmed

    return to_be_trimmed
