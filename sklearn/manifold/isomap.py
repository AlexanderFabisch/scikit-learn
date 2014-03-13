"""Isomap for manifold learning"""

# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD 3 clause (C) 2011

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist, squareform
from ..base import BaseEstimator, TransformerMixin
from ..neighbors import (NearestNeighbors, kneighbors_graph,
                         radius_neighbors_graph)
from ..utils import check_arrays
from ..utils.graph import graph_shortest_path
from ..decomposition import KernelPCA
from ..preprocessing import KernelCenterer
from ..linear_model import LinearRegression
from ..externals import six


def elbow_method(n_components, scores, elbow=True, verbose=0):
    """Elbow detection similar to DiffBIC knee detection.

    See http://www.cs.joensuu.fi/~zhao/PAPER/ICTAI08.pdf for details.

    Parameters
    ----------
    n_components : array-like, shape (n_component_samples,)
        Number of components where we have measured the score.

    scores : array-like, shape(n_component_samples,)
        Measured scores.

    elbow : bool, optional (default: True)
        Detect elbow. Otherwise we will detect a knee point.

    verbose : int, optional (default: 0)
        Verbosity level.
    """
    if elbow:
        # We want to recognize elbows, no knee points
        scores = 1 - np.asarray(scores)

    # Determine trend
    lin_model = LinearRegression().fit(
        np.asarray(n_components)[:, np.newaxis], scores)
    increasing_trend = lin_model.coef_[0] > 0
    if verbose:
        trend = "Increasing" if increasing_trend else "Decreasing"
        print("%s trend, coefficient of linear model is %.3f"
              % (trend, lin_model.coef_[0]))

    components_range = np.max(n_components) - np.min(n_components)
    scores_min = np.min(scores)
    scores_range = np.max(scores) - scores_min
    C1 = components_range * (scores - scores_min) / scores_range
    Cm = C1 / np.asarray(n_components)
    Cm_min = np.min(Cm)
    Cm_range = np.max(Cm) - Cm_min
    C2 = components_range * (Cm - Cm_min) / Cm_range
    if increasing_trend:
        diff_BIC = (C1 + C2) / 2
    else:
        diff_BIC = np.abs(C1 - C2) / 2

    n_opt_components = n_components[np.argmax(diff_BIC)]

    if elbow:
        diff_BIC = components_range - diff_BIC
    return diff_BIC, n_opt_components


class BaseIsomap(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for Isomap transformers."""

    @abstractmethod
    def __init__(self):
        pass

    def _init_params(self, n_components, eigen_solver, tol, max_iter,
                     path_method, neighbors_algorithm):
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm

    def _fit_nbrs(self, nbrs, X):
        X, = check_arrays(X, sparse_format='dense')
        nbrs.fit(X)
        self.training_data_ = nbrs._fit_X

    def _fit_kpca(self, kng):
        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)

        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter)
        self.embedding_ = self.kernel_pca_.fit_transform(G)

    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float

        Notes
        -------
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

    def _shortest_distance_to_training_data(self, n_samples, distances,
                                            indices):
        """Compute the graph of shortest distances to training data."""
        #Create the graph of shortest distances from X to self.training_data_
        # via the nearest neighbors of X.
        #This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:
        G_X = np.zeros((n_samples, self.training_data_.shape[0]))
        for i in range(n_samples):
            G_X[i] = np.min((self.dist_matrix_[indices[i]]
                             + distances[i][:, None]), 0)

        G_X **= 2
        G_X *= -0.5
        return G_X

    def residual_variance(self, n_components):
        """Compute residual variance of the embedding.

        The residual variance is defined as

        .. math::

            1 - R^2(D, D_Y)

        where :math:`D_Y` is the matrix of Euclidean distances in the
        low-dimensional embedding recovered by Isomap, :math:`D` is the
        distance matrix in the original space and R is the correlation
        coefficient.

        Parameters
        ----------
        n_components : int <= self.n_components
            Number of embedded components that we be considered.

        Returns
        -------
        residual_variance : float
        """
        if not n_components <= self.n_components:
            raise ValueError("n_components must be less than or equal %d but "
                             "is %d" % (self.n_components, n_components))

        embedded = (self.kernel_pca_.alphas_[:, :n_components] *
                    np.sqrt(self.kernel_pca_.lambdas_[:n_components]))
        D = pdist(self.training_data_, "euclidean")
        D_Y = pdist(embedded, "euclidean")
        return 1 - np.corrcoef(D, D_Y)[0, 1] ** 2


class Isomap(BaseIsomap, TransformerMixin):
    """Isomap Embedding based on k nearest neighbors aka k-Isomap

    Non-linear dimensionality reduction through Isometric Mapping

    Parameters
    ----------
    n_neighbors : integer
        number of neighbors to consider for each point.

    n_components : integer
        number of coordinates for the manifold

    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
            for the given problem.
        'arpack' : Use Arnoldi decomposition to find the eigenvalues
            and eigenvectors.
        'dense' : Use a direct solver (i.e. LAPACK)
            for the eigenvalue decomposition.

    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.
        'auto' : attempt to choose the best algorithm automatically
        'FW' : Floyd-Warshall algorithm
        'D' : Dijkstra algorithm with Fibonacci Heaps

    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    Attributes
    ----------
    `embedding_` : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    `kernel_pca_` : object
        `KernelPCA` object used to implement the embedding.

    `training_data_` : array-like, shape (n_samples, n_features)
        Stores the training data.

    `nbrs_` : sklearn.neighbors.NearestNeighbors instance
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    `dist_matrix_` : array-like, shape (n_samples, n_samples)
        Stores the geodesic distance matrix of training data.

    References
    ----------

    [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
        framework for nonlinear dimensionality reduction. Science 290 (5500)
    """

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto'):

        self.n_neighbors = n_neighbors
        self._init_params(n_components, eigen_solver, tol, max_iter,
                          path_method, neighbors_algorithm)

    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm)
        self._fit_nbrs(self.nbrs_, X)
        kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                               mode='distance')
        self._fit_kpca(kng)

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, precomputed tree, or NearestNeighbors
            object.

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)
        G_X = self._shortest_distance_to_training_data(X.shape[0], distances,
                                                       indices)
        return self.kernel_pca_.transform(G_X)


class EpsilonIsomap(BaseIsomap, TransformerMixin):
    """Isomap Embedding based on epsilon nearest neighbors aka epsilon-Isomap

    Non-linear dimensionality reduction through Isometric Mapping

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth`radius_neighbors`
        queries.

    n_components : integer
        number of coordinates for the manifold

    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
            for the given problem.
        'arpack' : Use Arnoldi decomposition to find the eigenvalues
            and eigenvectors.
        'dense' : Use a direct solver (i.e. LAPACK)
            for the eigenvalue decomposition.

    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.
        'auto' : attempt to choose the best algorithm automatically
        'FW' : Floyd-Warshall algorithm
        'D' : Dijkstra algorithm with Fibonacci Heaps

    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    Attributes
    ----------
    `embedding_` : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    `kernel_pca_` : object
        `KernelPCA` object used to implement the embedding.

    `training_data_` : array-like, shape (n_samples, n_features)
        Stores the training data.

    `nbrs_` : sklearn.neighbors.NearestNeighbors instance
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    `dist_matrix_` : array-like, shape (n_samples, n_samples)
        Stores the geodesic distance matrix of training data.

    References
    ----------

    [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
        framework for nonlinear dimensionality reduction. Science 290 (5500)
    """

    def __init__(self, radius=1.0, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto'):

        self.radius = radius
        self._init_params(n_components, eigen_solver, tol, max_iter,
                          path_method, neighbors_algorithm)

    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(radius=self.radius,
                                      algorithm=self.neighbors_algorithm)
        self._fit_nbrs(self.nbrs_, X)
        kng = radius_neighbors_graph(self.nbrs_, self.radius,
                                     mode='distance')
        self._fit_kpca(kng)

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, precomputed tree, or NearestNeighbors
            object.

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X: {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
        """
        distances, indices = self.nbrs_.radius_neighbors(
            X, return_distance=True)
        G_X = self._shortest_distance_to_training_data(X.shape[0], distances,
                                                       indices)
        return self.kernel_pca_.transform(G_X)

    def determine_connected_components(self):
        """Determine number of connected components.

        Returns
        -------
        prototypes : array, shape (n_samples,)
            Index of first sample from the corresponding component in the
            training set.

        n_connected_components : int
            Number of connected components, i.e. subgraphs in the training set
            that are not connected to each other.
        """
        self.dist_matrix_[self.dist_matrix_ == 0] = -1
        np.fill_diagonal(self.dist_matrix_, 0.0)
        prototypes = np.argmin(self.dist_matrix_ == -1, axis=0)
        n_connected_components = np.unique(prototypes).shape[0]
        return prototypes, n_connected_components
