from .config import *
from .class_defs import *


def block_sort(X: np.ndarray,
               row_partition: dict,
               col_partition: dict,
               row_labels: Union[list, None] = None,
               col_labels: Union[list, None] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, list, list], Tuple[np.ndarray, list]]:
    """ block_sort is a utility function that sorts a data matrix X according to given
        row and column partitions.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        row_partition : dict
            Dictionary with row numbers as keys and partition labels as values
        col_partition : dict
            Dictionary with column numbers as keys and partition labels as values
        row_labels : list (optional)
            List of row labels.
            defaults to :code:`None`.
        col_labels : list (optional)
            List of column labels
            defaults to :code:`None`.

        Returns
        -------
        X_sort : np.ndarray
            Data matrix sorted in blocks according to row_partition and col_partition.
        row_labels_sort : list (optional)
            List of sorted row labels, returned only if row_labels is not None.
        col_labels_sort : list (optional)
            List of sorted column labels, returned only if col_labels is not None.
    """

    n, p = X.shape
    g = len(set(row_partition.values()))
    m = len(set(col_partition.values()))
    X_sort = np.zeros([n, p])
    I_map = []
    J_map = []
    row_labels_sort = []
    col_labels_sort = []

    for k in range(g):
        I_map.extend([key1 for key1, val1 in row_partition.items() if val1 == k])

    for l in range(m):
        J_map.extend([key2 for key2, val2 in col_partition.items() if val2 == l])

    # row outliers are treated separately (we dump them after the last column group)
    I_map.extend([key1 for key1, val1 in row_partition.items() if val1 == -1])

    # column outliers are treated separately (we dump them after the last column group)
    J_map.extend([key2 for key2, val2 in col_partition.items() if val2 == -1])

    for i in range(n):
        if row_labels:
            row_labels_sort.append(row_labels[I_map[i]])
        for j in range(p):
            if J_map[j] == -1:
                # column marked as outlier, does not have a cluster membership
                pass
            else:
                X_sort[i, j] = X[I_map[i], J_map[j]]
    if col_labels:
        for j in range(p):
            col_labels_sort.append(col_labels[J_map[j]])

    if (row_labels is not None) and (col_labels is not None):
        return X_sort, row_labels_sort, col_labels_sort
    elif row_labels is not None:
        return X_sort, row_labels_sort
    elif col_labels is not None:
        return X_sort, col_labels_sort
    else:
        return X_sort


def generate_m(n: int,
               p: int,
               alpha_o: float,
               alpha_m: float,
               block: bool = False,
               Z: Union[np.ndarray, None] = None,
               W: Union[np.ndarray, None] = None,
               seed: Union[int, None] = None,
               rng: Union[None, np.random.Generator] = None) -> np.ndarray:
    """ generate_m is a utility function that can be used to simulate cellwise contamination and / or missingness by
    generating a mask matrix M, of the same size of the data matrix X to be contaminated.

    Parameters
    ----------
    n : int
        Number of rows of data matrix X
    p : int
        Number of columns of data matrix X
    alpha_o : float
        Fraction of outlying cells
    alpha_m : float
        Fraction of missing cells
    block : bool (optional)
        Whether the contamination rate is controlled block-wise (True) or globally (False)
        Defaults to :code:`False`
    Z : np.ndarray or None (optional)
        Row partition matrix, required if :code:`block=True` to impose equal block contamination
        defaults to :code:`None`
    W : np.ndarray or None (optional)
        Column partition matrix, required if :code:`block=True` to impose equal block contamination
        defaults to :code:`None`
    seed : int or None (optional)
        Seed for reproducibility
        defaults to :code:`None`
    rng : numpy.random.Generator (optional)
        Random number generator object to ensure reproducibility.
        This is the recommended way of ensuring reproducibility, since it avoids accidentally using the same seed
        from other parts of the code.
        If :code:`None`, then `seed` is used to initialise a new random number generator.
        Defaults to None.

    Returns
    -------
    M : np.ndarray
        Pattern matrix (M_ij = -1 if cell X_ij is missing, M_ij = +1 if outlying, M_ij = 0 otherwise)

    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)
    M = np.zeros((n, p))

    if block:
        g = Z.shape[1]
        m = W.shape[1]

        for k in range(g):
            for l in range(m):

                # get indices corresponding to block (k,l)
                I = np.where(Z[:, k])[0].tolist()
                J = np.where(W[:, l])[0].tolist()
                IJ = list(product(I, J))

                # number of cells to be contaminated (outliers and NaNs)
                n_o = int(np.floor(alpha_o * len(IJ)))
                n_m = int(np.floor(alpha_m * len(IJ)))

                # randomly select indices of cells in block (k,l) to be contaminated
                rand_idx = rng.choice(a=len(IJ), size=n_o + n_m, replace=False)
                IJ = np.array(IJ)
                rand_idx_o = IJ[rand_idx[0:n_o], :]
                rand_idx_m = IJ[rand_idx[n_o:n_o + n_m], :]

                # update mask matrix accordingly
                M[rand_idx_o[:, 0], rand_idx_o[:, 1]] = +1
                M[rand_idx_m[:, 0], rand_idx_m[:, 1]] = -1

    else:
        n_o = int(np.floor(alpha_o * n * p))
        n_m = int(np.floor(alpha_m * n * p))

        rand_idx = rng.choice(a=n * p, size=n_o + n_m, replace=False)
        rand_idx_o = rand_idx[0:n_o]
        rand_idx_m = rand_idx[n_o:n_o + n_m]

        M.ravel()[rand_idx_o] = +1
        M.ravel()[rand_idx_m] = -1

    return M


def generate_partitions(n: int,
                        p: int,
                        g: int,
                        m: int,
                        seed: int = 0,
                        rng: Union[None, np.random.Generator] = None) -> (dict, dict, np.ndarray, np.ndarray):
    """ generate_partitions is a utility function that can be used to simulate the row and column partitions of an LBM.

    Parameters
    ----------
    n : int
        Number of rows
    p : int
        Number of columns
    g : int
        Number of row groups
    m : int
        Number of column groups
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
    Tuple containing row and column partitions as dicts.

    """

    rng0 = np.random.default_rng(seed=0)
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    row_w = g * np.ones(g)
    row_partition_weights = rng0.dirichlet(row_w)
    col_w = m * np.ones(m)
    col_partition_weights = rng0.dirichlet(col_w)

    degenerate = True   # to avoid generating a partition with empty row or column groups
    while degenerate:
        row_partition_sizes = rng.multinomial(n, row_partition_weights)
        col_partition_sizes = rng.multinomial(p, col_partition_weights)
        degenerate = np.any(row_partition_sizes == 0) | np.any(col_partition_sizes == 0)

    row_labels = np.repeat(np.arange(len(row_partition_weights)), row_partition_sizes)
    col_labels = np.repeat(np.arange(len(col_partition_weights)), col_partition_sizes)

    rng.shuffle(row_labels)
    rng.shuffle(col_labels)

    row_partition = dict(zip(list(range(n)), row_labels))
    col_partition = dict(zip(list(range(p)), col_labels))

    return row_partition, col_partition, row_partition_weights, col_partition_weights


def get_block_values(X: np.ndarray,
                     Z: np.ndarray,
                     W: np.ndarray,
                     k: int,
                     l: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    get_block_values gets the cells in a given block of the data matrix X identified by (Z=k, W=l),
    Z and K being the row and column latent variables defining the partitions. Along the cells also
    their indices in X are returned.

    Parameters
    ----------
    X: np.ndarray
        Data matrix
    Z: np.ndarray
        Row partition matrix
    W: np.ndarray
        Column partition matrix
    k: int
        Row group label
    l: int
        Column group label

    Returns
    -------
    block_vals: np.ndarray
        Cells of X belonging to block (Z=k, W=l)
    IJ: np.ndarray
        Indices corresponding to the cells in block_vals

    """

    I = np.where(Z[:, k])[0].tolist()
    J = np.where(W[:, l])[0].tolist()
    IJ = np.array(list(product(I, J)))

    if IJ.size == 0:
        return np.array([]).reshape(len(I), len(J)), IJ

    block_vals = X[IJ[:, 0], IJ[:, 1]].reshape(len(I), len(J))

    return block_vals, IJ


def get_row_col_outliers(P: np.ndarray) -> list:
    """ get_outliers obtains row / column outliers from given partition matrix P

        Parameters
        ----------
        P : np.ndarray
            Binary matrix defining row / column partition
            (P[i,k]==1 iff row / column 'i' is in row / column group 'k')

        Returns
        -------
        l : list
            Indices of trimmed rows / columns

    """

    l = []
    for i, s in enumerate(np.sum(P, axis=1).tolist()):
        if s == 0:
            l.append(i)
    return l


def part_dict_to_matrix(d: dict) -> np.ndarray:
    """ part_dict_to_matrix is a utility function that maps a partition dict to a matrix.

    Parameters
    ----------
    d : dict
        Dictionary representation of a partition, i.e. dictionary of the form:
        {row/column : partition_label}
        Labels are assumed to be integers starting from 0
        Negative labels can be used to represent row/column outliers

    Returns
    -------
    P : np.ndarray
        Partition matrix (P[i,k] = 1 iff col/row i is in group k, else P[i,k] = 0)

    """

    n = len(d.keys())
    g = max(set(d.values())) + 1

    P = np.zeros((n, g))

    for i, key in enumerate(d):
        if d[key] >= 0:
            P[i, d[key]] = 1

    return P


def part_matrix_to_dict(P: np.ndarray,
                        exclude: Iterable = ()) -> dict:
    """ part_matrix_to_dict is a utility function that maps a partition matrix to a dict.

        Parameters
        ----------
        P : np.ndarray
            Partition matrix (P[i,k] = 1 iff col/row i is in group k, else P[i,k] = 0)
        exclude : Iterable (optional)
            Rows or columns to exclude from the output partition dict
            Defaults to the empty list

        Returns
        -------
        dict object having as keys the row/column indices and as values the corresponding
        group labels

    """

    labels_list = []
    for i in range(P.shape[0]):
        if i not in exclude:
            labels_list.append(np.nonzero(P[i, :])[0][0])
        else:
            labels_list.append(-1)

    return dict(zip(list(range(P.shape[0])), labels_list))


def partition_matrices_init(n: int,
                            p: int,
                            g: int,
                            m: int,
                            a: float,
                            b: float,
                            seed: int = 0,
                            rng: Union[None, np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ partition_matrices_init randomly initialises the row and column partition matrices
        (Z and W respectively).

        Parameters
        ----------
        n : int
            Number of rows of X
        p : int
            Number of columns of X
        g : int
            Number of row groups
        m : int
            Number of column groups
        a : float
            Trimming level on rows, must be in [0, 0.5)
        b : float
            Trimming level on columns, must be in [0, 0.5)
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
            n-by-g row partition matrix
        W : np.ndarray
            p-by-m column partition matrix

    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    Z = np.zeros((n, g))
    l_row = rng.integers(0, g, n)
    out_row = rng.choice(range(n), size=int(np.ceil(a * n)), replace=False)
    for i, position in enumerate(l_row):
        if i not in out_row:
            Z[i, position] = 1

    W = np.zeros((p, m))
    l_col = rng.integers(0, m, p)
    out_col = rng.choice(range(p), size=int(np.ceil(b * p)), replace=False)
    for i, position in enumerate(l_col):
        if i not in out_col:
            W[i, position] = 1

    return Z, W


def mixing_proportions(Z, W):
    """ mixing_proportions computes mixing proportions of row and column partitions.

    Input
    -----
    Z : np.ndarray
        n-by-g row partition matrix
    W : np.ndarray
        p-by-m column partition matrix

    Output
    ------
    Pi : np.ndarray
        Row mixing proportions
    Rho : np.ndarray
        Column mixing proportions

    """

    Pi = Z.sum(axis=0) / Z.sum()
    Rho = W.sum(axis=0) / W.sum()

    return Pi, Rho


def is_degenerate(params: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None] = None,
                  density: Union[str, None] = None,
                  Z: Union[np.ndarray, None] = None,
                  W: Union[np.ndarray, None] = None,
                  M: Union[np.ndarray, None] = None) -> bool:
    """ is_degenerate returns True if the parameters or partitions are degenerate, False otherwise.

        Parameters
        ----------
        params : np.ndarray, tuple of np.ndarrays, or None
            Parameters of block densities.
                - If :code:`density == "Poisson"`, Params must be a g-by-m np.ndarray;
                - If :code:`density == "Normal"`, Params must be a tuple of two g-by-m np.ndarrays;
                - If :code:`None`, the function checks the degeneracy of the partitions, which must be provided.
            Defaults to :code:`None`.
        density : str or None
            Either "Poisson" or "Normal", to specify model density.
            If :code:`None`, the function checks the degeneracy of the partitions, which must be provided.
            Defaults to :code:`None`.
        Z : np.ndarray or None
            Row partition matrix
        W : np.ndarray or None
            Column partition matrix
        M : np.ndarray or None
            Mask matrix

        Returns
        -------
        is_degenerate : bool
            :code:`True` if the given parameters are not admissible for the specified density, otherwise :code:`False`.

    """

    # checking degeneracy of partitions, if provided
    if (Z is not None) and (W is not None):
        if M is None:
            return not (np.all(Z.sum(axis=0)) and np.all(W.sum(axis=0)))
        else:
            # if a mask matrix M is passed, check that no block is entirely trimmed
            return not (np.all(Z.sum(axis=0)) and np.all(W.sum(axis=0)) and np.all(Z.T @ M @ W))

    # if Z and W are not provided, check degeneracy of parameters
    if isinstance(params, tuple) and density == "Normal":
        return not np.all(params[1])
    else:
        return not np.all(params)  # Poisson case
