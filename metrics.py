import numpy as np


def pe(u):
    """Compute the partition entroppy coefficient score with 1 the optimal value.

    Parameters
    ----------
    u : ndarray of shape (n_samples, n_clusters)
        The fuzzy partition.

    Returns
    -------
    pe_score : float
        The partition entropy score.
    """
    n_samples = u.shape[0]
    pe_score = np.sum(-u * np.log(u)).sum() / n_samples
    pe_score = round(pe_score, 2)
    return pe_score


def pc(u):
    """Compute the partition coefficient score with 0 the optimal value.

    Parameters
    ----------
    u : ndarray of shape (n_samples, n_clusters)
        The fuzzy partition.

    Returns
    -------
    pc_score : float
        The partition coefficient score.
    """
    n_samples = u.shape[0]
    pc_score = np.sum(u**2).sum() / n_samples
    pc_score = round(pc_score, 2)
    return pc_score
