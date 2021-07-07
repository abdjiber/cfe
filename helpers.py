import numpy as np
import warnings
from exceptions import CFEWarning


def check_params(cfe, X):
    """Check the correcteness of input parameters.

    Parameters
    ----------
    cfe : CFE
        A CFE instance.

    X : ndarray of shape (n_samples, n_features)
        The input intances to be clustered.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        If X contains features with one unique category the feature is dropped.
    """
    n_samples = X.shape[0]
    if cfe.n_clusters < 2:
        raise ValueError("n_clusters should be >= 2.")
    if cfe.n_clusters > n_samples:
        raise ValueError(f"n_clusters should be <= {n_samples}")
    if cfe.m < 1:
        raise ValueError("m should be > 1.")
    if cfe.alpha < 0:
        raise ValueError("alpha should be > 0.")
    if cfe.epsillon < 0:
        raise ValueError("epsillon should be > 0.")
    if not isinstance(cfe.verbose, bool):
        raise ValueError("verbose should be a Boolean.")
    attr_with_one_uniq_val = list()
    for l in range(X.shape[1]):
        _, uniq_vals = np.unique(X[:, l], return_counts=True)
        n_l = len(uniq_vals)
        if n_l == 1:
            attr_with_one_uniq_val.append(l)
    if attr_with_one_uniq_val:
        message = f"Attributes {attr_with_one_uniq_val} contain one unique\
            value,they will be dropped before training."

        warnings.warn(message, category=CFEWarning, stacklevel=0)
    X = np.delete(X, attr_with_one_uniq_val, axis=1)
    return X


def get_randn(n_l):
    """Generate random values.

    Parameters
    ----------
    n_l : int
        The length of random numbers to generate.

    Returns
    -------
    randn : array of shape n_l.
        The generated list of random numbers.
    """
    randn = np.abs(np.random.randn(n_l))
    randn /= np.sum(randn)
    return randn


def get_dom_vals_and_size(X):
    """Get the feature domains and size.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training instances to cluster.

    Returns
    -------
    dom_vals : array of shape n_unique_vals
        The domains of the features.

    n_attr_doms : int
        The length of the number of categories of X.
    """
    dom_vals = []
    n_attr_doms = []
    for k in range(X.shape[1]):
        unique = list(np.unique(X[:, k]))
        dom_vals += unique
        n_attr_doms += [len(unique)]
    return dom_vals, n_attr_doms


def exp_sum_u_ij_t(um, X, a_l, j, l, t, alpha):
    """Update the center for the jth cluster restricted to the lth feature.

    Paramaters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training instances to cluster.

    um : ndarray of shape (n_samples, n_clusters)
        The partition matrix power m.

    w_jl : ndarray
        The jth center restricted to the feature l.

    j : int
        The jth center to be updated.

    l : int
        The lth feature of the jth center to be updated.

    a_l : ndarray of shape (1, n_uniq_val)
        The domain of the lth feature.

    t : int
        The tth category of a_l.

    Returns
    -------
    exp_sum : float
        The weight of tth feature category.
        """
    n_samples = X.shape[0]
    freq = np.sum(um[np.array(X[:, l]) != a_l[t], j])
    exp_sum = np.exp(-freq / (n_samples * alpha))
    return exp_sum
