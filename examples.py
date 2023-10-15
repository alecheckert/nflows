import matplotlib.pyplot as plt
import numpy as np
from nflows import PlanarFlow, AffineFlow, Model
from nflows.optimize import Dataset, adam


def test0():
    d = 2
    n = 100
    model = Model([PlanarFlow(d) for i in range(10)])
    points = np.linspace(-1.0, 1.0, n)
    X = np.asarray(np.meshgrid(points, points)).reshape((2, n**2)).T
    Z = model.forward(X)
    LL = model.log_likelihood(X)
    L = np.exp(LL - LL.max())
    L /= L.sum()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].scatter(X[:, 0], X[:, 1], c="b", s=10, alpha=0.1)
    axes[1].scatter(Z[:, 0], Z[:, 1], c="b", s=10, alpha=0.1)
    axes[2].imshow(LL.reshape((n, n)), cmap="viridis")
    axes[3].imshow(L.reshape((n, n)), cmap="viridis")
    plt.tight_layout()
    plt.show()
    plt.close()


def test1():
    mean = np.array([2.0, 1.0])
    cov = np.array([[2.0, -1.0], [-1.0, 1.2]])
    model = Model(
        [AffineFlow(2), PlanarFlow(2), PlanarFlow(2), PlanarFlow(2), AffineFlow(2)]
    )
    # model = Model([AffineFlow(2)])

    class NormalDataset(Dataset):
        def __init__(self, mean, cov):
            self.mean = mean
            self.cov = cov

        @property
        def size(self) -> int:
            return 100000

        def get_batch(self, n: int) -> np.ndarray:
            return np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=n)

    dataset = NormalDataset(mean, cov)
    adam(model, dataset, "testmodel", epochs=2000, batch_size=100, verbose=True)

    points = np.linspace(-4.0, 4.0, 100)
    X = np.asarray(np.meshgrid(points, points)).reshape((2, 100**2)).T
    Xc = X - mean
    invC = np.linalg.inv(cov)
    LL0 = -0.5 * ((Xc @ invC.T) * Xc).sum(axis=1) - np.log(
        2 * np.pi * np.linalg.det(cov)
    )
    L0 = np.exp(LL0 - LL0.max())
    L0 /= L0.sum()
    LL = model.log_likelihood(X)
    L = np.exp(LL - LL.max())
    L /= L.sum()
    sample = model.generate(1000000)
    bins = np.linspace(-4.0, 4.0, len(points) + 1)
    H = np.histogram2d(sample[:, 1], sample[:, 0], bins=(bins, bins))[0].astype(
        np.float64
    )
    H /= H.sum()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(L0.reshape((100, 100)), cmap="viridis", origin="lower")
    axes[1].imshow(L.reshape((100, 100)), cmap="viridis", origin="lower")
    axes[2].imshow(H.reshape((100, 100)), cmap="viridis", origin="lower")
    plt.tight_layout()
    plt.show()
    plt.close()


def test2():
    n_components = 5
    N = 100000
    fractions = np.random.dirichlet(np.ones(n_components))
    means = []
    covs = []
    for i in range(n_components):
        means.append(np.random.normal(scale=2, size=2))
        # generate a random positive definite matrix
        c = np.random.normal(scale=2, size=2)
        cov = np.outer(c, c)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        boost = eigenvalues.min() + 0.1
        cov = eigenvectors @ np.diag(eigenvalues + boost) @ eigenvectors.T
        assert np.linalg.det(cov) > 0
        covs.append(cov)

    # Simulate data
    N_per_comp = np.random.multinomial(N, fractions)
    X = []
    for i in range(n_components):
        Ni = N_per_comp[i]
        Xi = np.random.multivariate_normal(mean=means[i], cov=covs[i], size=Ni)
        X.append(Xi)
    X = np.concatenate(X, axis=0)
    assert X.shape[0] == N

    # Make dataset
    class NormalDataset(Dataset):
        def __init__(self, X: np.ndarray):
            self.X = X

        @property
        def size(self) -> int:
            return self.X.shape[0]

        def get_batch(self, n: int) -> np.ndarray:
            indices = np.random.choice(np.arange(self.size), size=n, replace=False)
            return self.X[indices, :].copy()

    dataset = NormalDataset(X)
    model = Model(
        [
            AffineFlow(2),
            PlanarFlow(2),
            PlanarFlow(2),
            PlanarFlow(2),
            PlanarFlow(2),
            PlanarFlow(2),
            AffineFlow(2),
        ]
    )
    # model = Model([AffineFlow(2)])
    adam(model, dataset, "testmodel", epochs=3000, batch_size=1000, verbose=True)

    n_bins = 100
    vmin = -10.0
    vmax = 10.0
    points = np.linspace(vmin, vmax, n_bins)
    bins = np.linspace(vmin, vmax, n_bins + 1)
    grid = np.asarray(np.meshgrid(points, points)).reshape((2, n_bins**2)).T
    LL = model.log_likelihood(grid)
    L = np.exp(LL - LL.max())
    L /= L.sum()
    L = L.reshape((n_bins, n_bins))

    H0 = np.histogram2d(X[:, 1], X[:, 0], bins=(bins, bins))[0]

    try:
        Z = model.generate(N)
        H1 = np.histogram2d(Z[:, 1], Z[:, 0], bins=(bins, bins))[0]
    except Exception as exc:
        print(exc)
        H1 = np.zeros((n_bins, n_bins))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].imshow(H0, cmap="viridis", origin="lower")
    axes[1].imshow(L, cmap="viridis", origin="lower")
    axes[2].imshow(H1, cmap="viridis", origin="lower")
    plt.tight_layout()
    plt.show()
    plt.close()


def test3():
    """Uses the first energy function from Rezende &
    Mohamed 2015."""
    n = 1000
    dx = 0.01
    YX = np.indices((n, n)).astype(np.float32).reshape((2, n**2)).T
    YX -= float(n//2)
    YX *= dx

    def log_likelihood(YX: np.ndarray) -> np.ndarray:
        R = np.sqrt((YX**2).sum(axis=1))
        ll0 = -0.5 * ((R - 2) / 0.4)**2
        ll1 = np.log(np.exp(-0.5 * ((YX[:,0] - 2) / 0.6)**2) + np.exp(-0.5 * ((YX[:,0] + 2) / 0.6)**2))
        ll = ll0 + ll1

        return ll, ll0, ll1

    ll, term0, term1 = log_likelihood(YX)
    ll -= ll.max()
    p = np.exp(ll)
    p /= p.sum()
    p = p.reshape((n, n))

    entropy = -(p.ravel() * ll.ravel()).mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(term0.reshape((n, n)))
    axes[1].imshow(term1.reshape((n, n)))
    axes[2].imshow(p)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Via rejection sampling
    N = 10000000
    pup = p / p.max()
    yx = np.random.uniform(-dx * (n//2), +dx * (n//2), size=(N, 2))
    ll, _, _ = log_likelihood(yx)
    ll -= ll.max()
    l = np.exp(ll)
    take = np.random.random(size=N) <= l
    yx = yx[take, :]
    bins = 10 * dx * (np.arange(100).astype(np.float32) - float(n//20))
    h = np.histogram2d(yx[:,0], yx[:,1], bins=(bins, bins))[0].astype(np.float32)
    print(f"{yx.shape[0]}/{N} samples from rejection sampling")
    plt.imshow(h, vmin=0, vmax=h.max())
    plt.show()
    plt.close()

    # Make dataset
    class DatasetRezende1(Dataset):
        def __init__(self, X: np.ndarray):
            self.X = X

        @property
        def size(self) -> int:
            return self.X.shape[0]

        def get_batch(self, n: int) -> np.ndarray:
            indices = np.random.choice(np.arange(self.size), size=n, replace=False)
            return self.X[indices, :].copy()


    dataset = DatasetRezende1(yx)
    model = Model(
        [
            AffineFlow(2),
            PlanarFlow(2),
            PlanarFlow(2),
            AffineFlow(2),
        ]
    )
    # model = Model([AffineFlow(2)])
    adam(model, dataset, "testmodel", epochs=3000, batch_size=10000, verbose=True)

    n_bins = bins.shape[0]
    grid = np.asarray(np.meshgrid(bins, bins)).reshape((2, n_bins**2)).T
    LL = model.log_likelihood(grid)
    L = np.exp(LL - LL.max())
    L /= L.sum()
    L = L.reshape((n_bins, n_bins))

    H0 = np.histogram2d(yx[:, 1], yx[:, 0], bins=(bins, bins))[0]

    try:
        Z = model.generate(N)
        H1 = np.histogram2d(Z[:, 1], Z[:, 0], bins=(bins, bins))[0]
    except Exception as exc:
        print(exc)
        H1 = np.zeros((n_bins, n_bins))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].imshow(H0, cmap="viridis", origin="lower")
    axes[1].imshow(L, cmap="viridis", origin="lower")
    axes[2].imshow(H1, cmap="viridis", origin="lower")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # test0()
    # test1()
    # test2()
    test3()
