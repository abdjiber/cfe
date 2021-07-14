import warnings

import numpy as np
import pandas as pd

from exceptions import ModelNotFittedError
import helpers
import metrics


class CFE():
    """Categorical fuzzy entropy clustering.

    Parameters
    ----------
    n_clusters : int
        The number of desired clusters.

    m : float, defaut=1.2
        The fuzziness weighting exponent, should be > 1.

    alpha : float, default=1e-2
        The fuzzy entropy weighting coefficient.

    epsillon : float, default=1e-3
        Stop criteria.

    seed : int
        The random state seed.

    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    u : ndarray of shape (n_samples, n_clusters)
        The partition matrix after at the convergence.

    cluster_centers : ndarray of shape (n_attr_doms, n_clusters)
        Weights of features categories.

    n_iter : int
        The number of iterations at the convergence.

    history : dict
        History of the cost(inertia), shape (n_iter).

    scores : list of computed scores, shape (len(metrics))

    Raises
    ------
    ValueError : if the value of `n_clusters` is less than 2.
    ValueError : if the value of `m` is less than 1.

    Returns
    -------
    A CFE instance.

    Examples
    --------
        import pandas as pd
        from cfe import CFE
        soybean_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data")
        soybean_df.columns = [f"A{i}" for i in range(1, soybean_df.shape[1] + 1)]
        true_labels = soybean_df.A36.values # Last column corresponds to the objects classes.
        soybean_df = soybean_df.drop("A36", axis=1)
        X = soybean_df.values
        features = list(soybean_df)
        cfe = CFE(n_clusters=4, m=1.1, verbose=False)
        cfe.fit(X, features)
        print("Scores")
        print("Partition coefficient: ", cfe.pe)
        print("Partition entropy: ", cfe.pc)

    References
    ----------
    > A. J. Djiberou Mahamadou, V. Antoine, E. M. Nguifo and S. Moreno,
    "Categorical fuzzy entropy c-means" 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE),
    Glasgow, United Kingdom, 2020, pp. 1-6, doi: 10.1109/FUZZ48607.2020.9177666.
    > Abdoul Jalil Djiberou Mahamadou, Violaine Antoine, Engelbert Mephu Nguifo, Sylvain Moreno,
    “Apport de l'entropie pour les c-moyennes floues sur des données catégorielles”, EGC 2021, vol. RNTI-E-37.
    """
    def __init__(self,
                 n_clusters,
                 m=1.2,
                 alpha=1e-2,
                 epsillon=1e-3,
                 seed=None,
                 verbose=False):
        """Create a CFE instance model."""
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.epsillon = epsillon
        self.verbose = verbose
        self.seed = seed
        self._is_fitted = False

    def _init_centers(self):
        """Initialize the cluster centers."""
        w0 = np.zeros((np.sum(self.n_attr_doms), self.n_clusters),
                      dtype='float')
        np.random.seed(self.seed)
        for j in range(self.n_clusters):
            k = 0
            l = 0
            for n_l in self.n_attr_doms:
                l += n_l
                randn = helpers.get_randn(n_l)
                w0[k:l, j] = randn
                k = l
        return w0

    def _distance_objects_to_clusters(self, X, w):
        """Compute the distance between objects and clusters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        w : ndarray of shape (n_attr_doms, n_clusters)
            The centers of clusters.

        Returns
        -------
        dist : np.array
            The distances between objects and clusters.
        """
        dist = np.zeros((self.n_samples, self.n_clusters), dtype='float')
        for i in range(self.n_samples):
            xi = X[i]
            for j in range(self.n_clusters):
                sum_ = 0.0
                k = 0
                l = 0
                for x_l, n_l in zip(xi, self.n_attr_doms):
                    l += n_l
                    dom_val = np.array(self._dom_vals[k:l])
                    w_ = np.array(w[k:l, j])
                    sum_ += 1 - np.sum(w_[dom_val == x_l])
                    k += n_l
                dist[i, j] = sum_
        return dist

    def _update_u(self, dist):
        """Update the partition matrix given the distances between objects and cluster centers."""
        u = np.zeros((self.n_samples, self.n_clusters), dtype='float')
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
                if 0 in dist[i, :]:
                    u[i, :] = 0
                    u[i, dist[i, :].tolist().index(0)] = 1
                    break
                else:
                    u[i, j] = np.sum(
                        (dist[i, j] / dist[i, :])**(2.0 / (self.m - 1)))**-1.0
        u = np.where(
            u > np.finfo(float).eps, u,
            0)  # Setting all values less than the floatting precision to 0.
        return u

    def _cost(self, u, d, w):
        """Compute the cost (intertia) from an iteration.

        Parameters
        ----------
        u : ndarray of shape (n_samples, n_clusters)
            The partition matrix.

        d : ndarray of shape (n_samples, n_clusters)
            The distance between objects and clusters.

        w : ndarray of shape (n_attr_doms, n_clusters)
            The centers of clusters.

        Returns
        -------
        cost : float
            The cost of an iteration.
        """
        cost_base = np.sum(u**self.m * d**2)
        entropy = np.sum(w * np.log(w))
        cost = cost_base + self.n_samples * self.alpha * entropy
        cost = np.round(cost, 3)
        return cost

    def _update_cluster_center(self, X, um, w_jl, j, l, a_l, n_l):
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

        n_l : int
            The length of a_l.

        Returns
        -------
        w_jl : ndarray
            The updated jth center restricted to the lth feature.
        """
        sum_u_ij = 0.0
        list_ = list()
        for t in range(n_l):
            sum_u_ij += helpers.exp_sum_u_ij_t(um, X, a_l, j, l, t, self.alpha)
            list_.append(helpers.exp_sum_u_ij_t(um, X, a_l, j, l, t, self.alpha))
        for t in range(n_l):
            w_jl[t] = helpers.exp_sum_u_ij_t(um, X, a_l, j, l, t,
                                     self.alpha) / sum_u_ij
        return w_jl

    def _update_cluster_centers(self, X, u):
        """Update the cluster centers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        u : ndarray of shape (n_samples, n_clusters)
            The partition matrix.

        Returns
        -------
        w : ndarray of shape (n_attr_doms, n_clusters)
            The updated cluster centers.
        """
        um = u**self.m
        w = np.zeros((sum(self.n_attr_doms), self.n_clusters), dtype="float")
        for j in range(self.n_clusters):
            s = 0
            m = 0
            for l, n_l in enumerate(self.n_attr_doms):
                s += n_l
                w_jl = w[m:s, j]
                a_l = self._dom_vals[m:s]
                w[m:s,
                  j] = self._update_cluster_center(X, um, w_jl, j, l, a_l, n_l)
                m = s
        return w

    def fit(self, X, features):
        """Fit the CFE model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster.

        features : array of shape n_features
            The features names.

        Returns
        -------
        self
            Fitted model.
        """
        self.attr_names = features
        X = helpers.check_params(self, X)
        self.n_samples, self.n_features = X.shape
        dom_vals, attr_dom_n_l = helpers.get_dom_vals_and_size(X)
        self._dom_vals = dom_vals
        self.n_attr_doms = attr_dom_n_l
        old_cost = np.inf
        w0 = self._init_centers()
        w = w0
        n_iter = 0
        costs = list()
        not_finished = True
        while not_finished:
            dist = self._distance_objects_to_clusters(X, w)
            u = self._update_u(dist)
            w = self._update_cluster_centers(X, u)
            new_cost = self._cost(u, dist, w)
            not_finished = np.abs(new_cost - old_cost) > self.epsillon
            if self.verbose:
                print(f"Iter {n_iter + 1} cost {new_cost}")
            costs.append(new_cost)
            n_iter += 1
            if new_cost > old_cost:
                break
            old_cost = new_cost
        self._is_fitted = True
        columns = [f'C{i + 1}' for i in range(self.n_clusters)]
        self.u = pd.DataFrame(u, columns=columns)
        self.crisp_labels = np.argmax(u, axis=1)
        self.cluster_centers = pd.DataFrame(w, columns=columns)
        self.history = costs
        self.n_iter = n_iter
        self.pe = metrics.pe(self.u)
        self.pc = metrics.pc(self.u)
        return self

    def predict(self, X):
        """Perfom a prediction new objects intances.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
            The new objects instances to be clustered.

        Returns
        -------
        u : array of shape (n_samples, n_clusters)
            The predicted fuzzy partition of the new objects intances.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Please fit the model before using the predict method.")
        self.n_samples = X.shape[0]
        dist = self._distance_objects_to_clusters(X, self.cluster_centers.values)
        u = self._update_u(dist)
        columns = [f'C{i + 1}' for i in range(self.n_clusters)]
        u = pd.DataFrame(u, columns=columns)
        return u
