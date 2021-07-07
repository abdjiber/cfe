import numpy as np
import warnings
from exceptions import CFEWarning


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
