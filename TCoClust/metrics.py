"""
Module for evaluation metrics related to co-clustering models.

This module provides functions to assess the quality and accuracy of co-clustering
results, including the Co-clustering Adjusted Rand Index (CARI), parameter comparison,
mask comparison, and association measures such as the Phi-squared coefficient.

Main Components:
----------------
- cari(z1, w1, z2, w2):
    Computes the Co-clustering Adjusted Rand Index between two co-clusterings,
    ignoring rows and columns labeled as -1 (outliers).

- compare_m(M1, M2):
    Computes the disagreement between two mask matrices, useful for evaluating
    cellwise outlier detection.

- compare_parameters(set1, set2, norm=2):
    Computes a norm-based difference between two sets of Latent Block Model (LBM)
    block parameters, after sorting to align clusters.

- phi_squared(X, Z, W, M=None):
    Computes the Phi-squared coefficient to measure association strength between
    row and column partitions of a data matrix, optionally accounting for masked values.

Dependencies:
-------------
- Custom imports from the TCoClust package (may include third-party transitive dependencies):
    - `tcc.config`

"""

from .config import *


def cari(z1: list,
         w1: list,
         z2: list,
         w2: list) -> float:
    """ cari computes the Co-clustering Adjuster Rand Index (CARI).

    See: Robert et al., 2021, "Comparing high-dimensional partitions with the Co-clustering Adjusted Rand Index".

    Parameters
    ----------
    z1 : list
        Row labels of first co-clustering
    w1 : list
        Column labels of first co-clustering
    z2 : list
        Row labels of second co-clustering
    w2 : list
        Column labels of second co-clustering

    Returns
    -------
    CARI : float
        Co-clustering Adjuster Rand Index for the two co-clusterings specified by the input row and column labels

    Notes
    -----
    The value -1 is conventionally used to indicate outlier rows or columns in cluster assignments, which are excluded
    from the contingency tables and subsequent CARI computation to focus the metric on the agreement between true
    cluster assignments only.

    """

    # contingency tables, ignoring possible -1 labels (row or column outliers)
    _, CZ = crosstab(z1, z2, levels=[list(set(np.unique(z1)) - {-1}), list(set(np.unique(z2)) - {-1})])
    _, CW = crosstab(w1, w2, levels=[list(set(np.unique(w1)) - {-1}), list(set(np.unique(w2)) - {-1})])
    CZW = np.kron(CZ, CW)

    # grand sum
    S = np.sum([[comb(CZW[p, q], 2) for q in range(CZW.shape[1])] for p in range(CZW.shape[0])])

    # marginal sums
    SP = np.sum([comb(CZW.sum(axis=1)[p], 2) for p in range(CZW.shape[0])])
    SQ = np.sum([comb(CZW.sum(axis=0)[q], 2) for q in range(CZW.shape[1])])

    # numerator and denominator of CARI
    NUM = S - SP * SQ / comb(len(z1) * len(w1), 2)
    DEN = (SP + SQ) / 2 - SP * SQ / comb(len(z1) * len(w1), 2)

    # returning CARI
    return NUM / DEN


def compare_m(M1: np.ndarray, M2: np.ndarray) -> float:
    """ compare_m compares two mask matrices M1 and M2, computing the fraction of cells for which the two mask matrices
    disagree.

    Parameters
    ----------
    M1 : np.ndarray
        First mask matrix, must be a 2x2 binary Numpy ndarray (either of boolean or numeric type)
    M2 : np.ndarray
        Second mask matrix, must have the same shape of M1 and be a 2x2 binary Numpy ndarray (either of boolean or
        numeric type)

    Returns
    -------
    err : float
        Fraction of cells for which the two mask matrices disagree.
        In a simulation setting, if either M1 or M2 encodes the "real" outliers, then err is the misclassification rate.

    """

    err = np.sum(M1 != M2) / M1.size

    return err


def compare_parameters(set1: tuple,
                       set2: tuple,
                       norm: Union[int, str, None] = 2) -> tuple:
    """ compare_parameters is a utility function to compare two sets of LBM parameters, computing a chosen norm between
    the differences between the two sorted parameter sets, set1 and set2.

    Parameters
    ----------
    set1 : tuple
        First set of parameters. Each component of the tuple must be a Numpy ndarray representing one of the parameters
        of an LBM. For example, in a Poisson LBM, set1[0] and set1[1] contain, respectively, the row and column mixture
        proportions as vectors, and set1[2] contains a matrix representing the block parameters.
    set2 : tuple
        Second set of parameters. Must have the same length of set1 and contain parameters that are comparable to those
        contained in set1.
    norm : int, str or None (optional)
        Type of norm used to compare parameters (see documentation for numpy.linalg.norm()).
        Defaults to the 2-norm.

    Returns
    -------
    A tuple containing the differences in norm between the model parameters in set1 and set2.

    """

    norms = []
    for parameter1, parameter2 in zip(set1, set2):
        parameter1_s = np.sort(parameter1.ravel()).reshape(parameter1.shape)
        parameter2_s = np.sort(parameter2.ravel()).reshape(parameter2.shape)

        norms.append(np.linalg.norm(parameter1_s - parameter2_s, ord=norm))

    return tuple(norms)


def phi_squared(X: np.ndarray,
                Z: np.ndarray,
                W: np.ndarray,
                M: Union[np.ndarray, None] = None) -> float:
    """ phi_squared computes the Phi-squared coefficient, a measure of association here defined for the distribution
    induced by the row and column partitions on X by the partition matrices Z and W. Larger values indicate stronger
    association between the row and the column partitions.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    Z : np.ndarray
        Row partition matrix
    W : np.ndarray
        Column partition matrix
    M : np.ndarray or None (optional)
        Mask matrix
        defaults to :code:`None`

    Returns
    -------
    phi2 : float
        Phi-squared coefficient for the given input

    """

    if M is None:
        P = np.linalg.multi_dot([Z.T, X, W])
        N = P.sum()
        P = P / N
        P_k = P.sum(axis=1)
        P_l = P.sum(axis=0)
        P_k_P_l = np.outer(P_k, P_l)

        # phi2 = ((P - P_k_P_l)**2 / P_k_P_l).sum()
        # avoid division by zero when there are empty classes
        phi2 = np.divide((P - P_k_P_l) ** 2, P_k_P_l, out=np.zeros_like(P), where=(P_k_P_l != 0)).sum()
    else:
        P = np.linalg.multi_dot([Z.T, M * np.nan_to_num(X), W])
        N = P.sum()
        P = P / N
        P_k = P.sum(axis=1)
        P_l = P.sum(axis=0)
        P_k_P_l = np.outer(P_k, P_l)

        # phi2 = ((P - P_k_P_l)**2 / P_k_P_l).sum()
        # avoid division by zero when there are empty classes
        phi2 = np.divide((P - P_k_P_l) ** 2, P_k_P_l, out=np.zeros_like(P), where=(P_k_P_l != 0)).sum()

    return phi2
