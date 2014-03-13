"""
============================
Dataset Analysis with Isomap
============================

An illustration of how we can use Isomap to find some intrinsic properties of
datasets. These properties are the dimensionality of the data and the number of
components that compose the dataset.
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold.isomap import EpsilonIsomap, elbow_method
from sklearn.utils import check_random_state


def create_2_charts(n_samples=500, n_dims=6, noise_range=0.05, threshold=0.5,
                    offset=np.array([0.0, -0.5, 0.5, 0.2, -0.3, 0.3]),
                    random_state=None):
    random_state = check_random_state(random_state)
    X = np.empty((n_samples, n_dims))
    funcs = [np.sin, np.cos, np.tanh, np.log1p, np.sin]
    X[:, 0] = random_state.rand(n_samples)
    for d in range(n_dims - 1):
        X[:, d + 1] = funcs[d](2 * np.pi * X[:, 0])
    X += random_state.rand(n_samples, n_dims) * noise_range * 2 - noise_range
    X[np.where(X[:, 0] > threshold)] += offset[:n_dims]
    return X


X = create_2_charts(random_state=0)

isomap = EpsilonIsomap(n_components=6, radius=0.2).fit(X)

n_components = range(1, 7)
R = np.array([isomap.residual_variance(n) for n in n_components])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components, R)
plt.xlabel("Number of Components")
plt.ylabel("Residual Variance")

C, n_opt_components = elbow_method(n_components, R)
plt.subplot(1, 2, 2)
plt.title("Optimum Number of Components: %d" % n_opt_components)
plt.plot(n_components, C)
plt.xlabel("Number of Components")
plt.ylabel("DiffBIC of Residual Variance")

isomap = EpsilonIsomap(n_components=n_opt_components, radius=0.2)
Y_isomap = isomap.fit_transform(X)

n_dims = X.shape[1]
plt.figure(figsize=(n_dims * 2, 10))
colors = ["r", "g", "b", "k", "y", "orange"]
for d in range(n_dims):
    ax = plt.subplot(2, n_dims / 2, d + 1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_%d$" % (d + 1))
    plt.scatter(X[:, 0], X[:, d].ravel(), color=colors[d])
    plt.setp(ax, xticks=(), yticks=())

first, n_connected_components = isomap.determine_connected_components()
_, first_indices = np.unique(first, return_inverse=True)

plt.figure()
plt.title("Number of Connected Components: %d" % n_connected_components)
plt.scatter(X[:, 0].ravel(), Y_isomap[:, 0].ravel(), c=first_indices)
plt.show()
