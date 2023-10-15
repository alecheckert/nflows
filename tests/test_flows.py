import numpy as np
import unittest

from nflows.constants import DTYPE, EPSILON
from nflows.flows import AffineFlow, PlanarFlow, Permutation, RadialFlow, FLOWS
from nflows.model import Model
from nflows.utils import finite_differences, numerical_jacdet


class TestPlanarFlow(unittest.TestCase):
    def setUp(self):
        self.d = 4
        self.N = 10
        np.random.seed(666)
        self.X = np.random.normal(size=(self.N, self.d)).astype(DTYPE)

    def test_from_shape(self):
        d = self.d
        w = np.random.normal(size=d).astype(DTYPE)
        v = np.random.normal(size=d).astype(DTYPE)
        b = np.random.normal(size=1).astype(DTYPE)
        params = np.concatenate([w, v, b])
        flow = PlanarFlow.from_shape((d,), params)
        assert flow.d == d
        np.testing.assert_allclose(flow.w, w, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(flow.v, v, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(flow.b, b, atol=1e-6, rtol=1e-6)

    def test_forward(self):
        """Does-it-run? type test"""
        flow = PlanarFlow(self.d)
        Y, detjac = flow.forward(self.X)
        assert Y.shape == (self.N, self.d)
        assert detjac.shape == (self.N,)

    def test_invert(self):
        flow = PlanarFlow(self.d)
        X = self.X
        Y, _ = flow.forward(X)
        invY = flow.invert(Y)
        np.testing.assert_allclose(invY, X, atol=1e-5, rtol=1e-5)

    def test_parameters(self):
        flow = PlanarFlow(self.d)
        pars = flow.parameters
        assert pars["w"] is flow.w
        assert pars["v"] is flow.v
        assert pars["b"] is flow.b
        # numpy arrays are references
        pars["w"][0] += 1.0
        assert abs(pars["w"][0] - flow.w[0]) < 1e-6

    def test_jacdet(self):
        flow = PlanarFlow(self.d)

        # Values of X to test. For each of these, we
        # compare the analytical derivatives of the log
        # Jacobian determinant with respect to numerical
        # results from finite differences.
        sample_X = [
            np.array([[1.0, -2.0, 0.0, 0.5]]),
            np.array([[0.0, 0.0, -4.0, -2.0]]),
            np.array([[1.0, 1.0, 1.0, 1.0]]),
            np.array([[-0.5, 0.5, -0.5, 0.5]]),
        ]
        delta = 1e-4

        for X in sample_X:

            def f(X: np.ndarray) -> float:
                _, detjac = flow.forward(X)
                return np.log(detjac[0] + EPSILON)

            _, dlogdetjac_dX, _ = flow.backward(X, np.zeros_like(X))
            dlogdetjac_dX_num = finite_differences(f, X, delta=delta)
            np.testing.assert_allclose(
                dlogdetjac_dX, dlogdetjac_dX_num, atol=1e-4, rtol=1e-3
            )

    def test_backward(self):
        w = np.random.normal(size=self.d).astype(DTYPE)
        v = np.random.normal(size=self.d).astype(DTYPE)
        b = np.random.normal(size=1).astype(DTYPE)
        flow = PlanarFlow(d=self.d, w=w, v=v, b=b)

        def loss(X: np.ndarray) -> float:
            Y, detjac = flow.forward(X)
            return 0.5 * (Y**2).sum()

        X = self.X

        # For each datum, check that numerical differentiation
        # agrees with the analytical differentiation via the
        # Flow.backward routine
        for i in range(X.shape[0]):
            Xi = np.array([X[i, :]])

            # Evaluate partial derivatives of loss w.r.t. X
            # via numerical differentiation
            delta = 1e-3
            dL_dX_num = finite_differences(loss, Xi, delta=delta)

            # Evaluate via backpropagation
            Y, _ = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY, normalize=False)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)

    def test_backward_full_loss(self):
        """Test PlanarFlow.backward with a full normalizing flow
        loss, incorporating partial derivatives with respect to
        the log Jacobian determinant."""
        w = np.random.normal(size=self.d).astype(DTYPE)
        v = np.random.normal(size=self.d).astype(DTYPE)
        b = np.random.normal(size=1).astype(DTYPE)
        flow = PlanarFlow(d=self.d, w=w, v=v, b=b)

        def loss(X: np.ndarray) -> float:
            Y, detjac = flow.forward(X)
            L = 0.5 * (Y**2).sum() + (self.d / 2) * np.log(2 * np.pi)
            logdetjac = np.log(np.abs(detjac) + EPSILON)
            L -= logdetjac
            return L.sum()

        X = self.X

        # For each datum, check that numerical differentiation
        # agrees with the analytical differentiation via the
        # Flow.backward routine
        for i in range(X.shape[0]):
            Xi = np.array([X[i, :]])

            # Evaluate partial derivatives of loss w.r.t. X
            # via numerical differentiation
            delta = 1e-4
            dL_dX_num = finite_differences(loss, Xi, delta=delta)

            # Evaluate via backpropagation
            Y, _ = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)

    def test_gradient(self):
        w = np.random.normal(size=self.d).astype(DTYPE)
        v = np.random.normal(size=self.d).astype(DTYPE)
        b = np.random.normal(size=1).astype(DTYPE)

        n = 10
        X = np.random.normal(size=(n, self.d)).astype(DTYPE)

        def loss(pars: np.ndarray) -> float:
            w = pars[: self.d]
            v = pars[self.d : 2 * self.d]
            b = pars[2 * self.d :]
            flow = PlanarFlow(d=self.d, w=w, v=v, b=b)
            Y, detjac = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (self.d / 2) * np.log(2 * np.pi)
            dL_dY = Y.copy()
            logdetjac = np.log(np.abs(detjac) + EPSILON)
            L -= logdetjac
            return L.mean()

        pars = np.concatenate([w, v, b])
        delta = 1e-4
        dL_dpars_num = finite_differences(loss, pars, delta=delta)

        # Analytical loss
        flow = PlanarFlow(d=self.d, w=w, v=v, b=b)
        Y, _ = flow.forward(X)
        L = 0.5 * (Y**2).sum(axis=1) + (self.d / 2) * np.log(2 * np.pi)
        dL_dY = Y.copy()
        _, _, dL_dpars_ana = flow.backward(X, dL_dY)

        dL_dpars_ana = np.concatenate(
            [dL_dpars_ana["w"], dL_dpars_ana["v"], dL_dpars_ana["b"]]
        )
        np.testing.assert_allclose(dL_dpars_ana, dL_dpars_num, atol=1e-3, rtol=1e-3)


class TestAffineFlow(unittest.TestCase):
    def setUp(self):
        self.d = 4
        np.random.seed(666)
        self.mean = np.random.normal(size=self.d).astype(DTYPE)
        self.scale = np.random.normal(size=self.d).astype(DTYPE)

    def test_from_shape(self):
        d = self.d
        params = np.concatenate([self.mean, self.scale])
        flow = AffineFlow.from_shape((d,), params)
        assert flow.d == self.d
        np.testing.assert_allclose(flow.mean, self.mean, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(flow.scale, self.scale, atol=1e-6, rtol=1e-6)

    def test_parameters(self):
        mean = self.mean
        scale = self.scale
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)
        pars = flow.parameters
        assert pars["mean"] is flow.mean
        assert pars["scale"] is flow.scale
        np.testing.assert_allclose(pars["mean"], flow.mean, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(pars["scale"], flow.scale, atol=1e-5, rtol=1e-5)

    def test_invert(self):
        mean = self.mean
        scale = self.scale
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)
        X = np.random.normal(size=(10, self.d)).astype(DTYPE)
        Y, _ = flow.forward(X)
        Z = flow.invert(Y)
        np.testing.assert_allclose(Z, X, atol=1e-4, rtol=1e-4)

    def test_backward(self):
        mean = self.mean
        scale = self.scale
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)

        def loss(X: np.ndarray) -> float:
            Y, _ = flow.forward(X)
            return 0.5 * (Y**2).sum()

        X = np.random.normal(size=(10, self.d)).astype(DTYPE)

        # For each datum, check that numerical differentiation
        # agrees with the analytical differentiation via the
        # Flow.backward routine
        for i in range(X.shape[0]):
            Xi = np.array([X[i, :]])

            # Evaluate partial derivatives of loss w.r.t. X
            # via numerical differentiation
            delta = 1e-3
            dL_dX_num = finite_differences(loss, Xi, delta=delta)

            # Evaluate via backpropagation
            Y, _ = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY, normalize=False)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)

    def test_backward_full_loss(self):
        mean = self.mean
        scale = self.scale
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)

        def loss(X: np.ndarray) -> float:
            Y, detjac = flow.forward(X)
            L = 0.5 * (Y**2) + (self.d / 2) * np.log(2 * np.pi)
            logdetjac = np.log(np.abs(detjac) + EPSILON)
            L -= logdetjac
            return L.sum()

        X = np.random.normal(size=(10, self.d)).astype(DTYPE)

        # For each datum, check that numerical differentiation
        # agrees with the analytical differentiation via the
        # Flow.backward routine
        for i in range(X.shape[0]):
            Xi = np.array([X[i, :]])

            # Evaluate partial derivatives of loss w.r.t. X
            # via numerical differentiation
            delta = 1e-3
            dL_dX_num = finite_differences(loss, Xi, delta=delta)

            # Evaluate via backpropagation
            Y, _ = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)

    def test_jacdet(self):
        mean = self.mean
        scale = self.scale
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)
        n = 3
        X = np.random.normal(size=(1, self.d)).astype(DTYPE)
        _, jd = flow.forward(X)
        _, dlogdetjac_dX, _ = flow.backward(X, np.zeros_like(X))
        assert jd.shape == (X.shape[0],)
        np.testing.assert_allclose(jd, np.prod(scale), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(dlogdetjac_dX, 0.0, atol=1e-4, rtol=1e-4)

    def test_gradient(self):
        mean = self.mean
        scale = self.scale
        n = 10
        X = np.random.normal(size=(n, self.d)).astype(DTYPE)

        def loss(pars: np.ndarray) -> float:
            mean = pars[: self.d]
            scale = pars[self.d :]
            flow = AffineFlow(d=self.d, mean=mean, scale=scale)
            Y, detjac = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (self.d / 2) * np.log(2 * np.pi)
            dL_dY = Y.copy()
            logdetjac = np.log(np.abs(detjac) + EPSILON)
            L -= logdetjac
            return L.mean()

        pars = np.concatenate([mean, scale])
        delta = 1e-3
        dL_dpars_num = finite_differences(loss, pars, delta=delta)

        # Analytical loss
        flow = AffineFlow(d=self.d, mean=mean, scale=scale)
        Y, _ = flow.forward(X)
        L = 0.5 * (Y**2).sum(axis=1) + (self.d / 2) * np.log(2 * np.pi)
        dL_dY = Y.copy()
        _, _, dL_dpars_ana = flow.backward(X, dL_dY)
        dL_dpars_ana = np.concatenate([dL_dpars_ana["mean"], dL_dpars_ana["scale"]])
        np.testing.assert_allclose(dL_dpars_ana, dL_dpars_num, atol=1e-3, rtol=1e-3)


class TestPermutation(unittest.TestCase):
    def test_permutation(self):
        d = 4
        N = 10
        order = np.array([3, 1, 0, 2])
        X = np.random.randint(-10, 10, size=(N, d)).astype(DTYPE)
        flow = Permutation(d, order=order)
        Y, detjac = flow.forward(X)
        np.testing.assert_allclose(X[:, 3], Y[:, 0], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(X[:, 1], Y[:, 1], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(X[:, 0], Y[:, 2], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(X[:, 2], Y[:, 3], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            np.abs(detjac), np.ones(N, dtype=DTYPE), atol=1e-6, rtol=1e-6
        )
        Z = flow.invert(Y)
        np.testing.assert_allclose(Z, X, atol=1e-6, rtol=1e-6)
        dL_dY = np.random.randint(-10, 10, size=(N, d)).astype(DTYPE)
        dL_dX, dlogdetjac_dX, dL_dpars = flow.backward(X, dL_dY)
        np.testing.assert_allclose(dL_dX[:, 3], dL_dY[:, 0], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(dL_dX[:, 1], dL_dY[:, 1], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(dL_dX[:, 0], dL_dY[:, 2], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(dL_dX[:, 2], dL_dY[:, 3], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            dlogdetjac_dX, np.zeros((N, d), dtype=DTYPE), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            dL_dpars["order"], np.zeros(d, dtype=DTYPE), atol=1e-6, rtol=1e-6
        )


class TestRadialFlow(unittest.TestCase):
    def setUp(self):
        np.random.seed(666)
        self.N = 10
        self.d = 4
        self.b = np.random.normal(size=self.d).astype(DTYPE)
        self.alpha = np.random.normal(size=1).astype(DTYPE)
        self.beta = np.random.normal(size=1).astype(DTYPE)
        self.X = np.random.normal(size=(self.N, self.d)).astype(DTYPE)

    def test_backward_max_likelihood(self):
        """Test accuracy of backpropagated partial derivatives
        without the log Jacobian determinant penalty terms
        (corresponding to a negative log likelihood loss in
        the latent space)."""
        N = self.N
        d = self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        X = self.X
        flow = RadialFlow(d, b=b, alpha=alpha, beta=beta)
        Y, _ = flow.forward(X)
        dL_dY = Y.copy() / N

        def loss(X: np.ndarray) -> float:
            Y, _ = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)
            return L.mean()

        delta = 1e-4
        dL_dX_num = finite_differences(loss, X, delta=delta)
        dL_dX_ana, _, _ = flow.backward(X, dL_dY)
        np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-5, rtol=1e-5)

    def test_beta_gradient(self):
        """Test that RadialFlow accurately computes the gradient
        with respect to the beta parameter."""
        N = self.N
        d = self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        X = self.X
        flow = RadialFlow(d, b=b, alpha=alpha, beta=beta)
        Y, _ = flow.forward(X)
        dL_dY = Y.copy()
        _, _, dL_dpars = flow.backward(X, dL_dY)
        dL_dbeta_ana = dL_dpars["beta"]

        def loss(beta: np.ndarray) -> float:
            flow.beta = beta
            Y, _ = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)
            return L.mean()

        delta = 1e-4
        dL_dbeta_num = finite_differences(loss, beta, delta=delta)
        np.testing.assert_allclose(dL_dbeta_ana, dL_dbeta_num, atol=1e-5, rtol=1e-5)

    def test_alpha_gradient(self):
        """Test that RadialFlow accurately computes the gradient
        with respect to the alpha parameter."""
        N = self.N
        d = self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        X = self.X
        flow = RadialFlow(d, b=b, alpha=alpha, beta=beta)
        Y, _ = flow.forward(X)
        dL_dY = Y.copy()
        _, _, dL_dpars = flow.backward(X, dL_dY)
        dL_dalpha_ana = dL_dpars["alpha"]

        def loss(alpha: np.ndarray) -> float:
            flow.alpha = alpha
            Y, _ = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)
            return L.mean()

        delta = 1e-4
        dL_dalpha_num = finite_differences(loss, alpha, delta=delta)
        np.testing.assert_allclose(dL_dalpha_ana, dL_dalpha_num, atol=1e-5, rtol=1e-5)

    def test_b_gradient(self):
        """Test that RadialFlow accurately computes the gradient
        with respect to the b parameter."""
        N = self.N
        d = self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        X = self.X
        flow = RadialFlow(d, b=b, alpha=alpha, beta=beta)
        Y, _ = flow.forward(X)
        dL_dY = Y.copy()
        _, _, dL_dpars = flow.backward(X, dL_dY)
        dL_db_ana = dL_dpars["b"]

        def loss(b: np.ndarray) -> float:
            flow.b = b
            Y, _ = flow.forward(X)
            L = 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)
            return L.mean()

        delta = 1e-4
        dL_db_num = finite_differences(loss, b, delta=delta)
        np.testing.assert_allclose(dL_db_ana, dL_db_num, atol=1e-5, rtol=1e-5)


class TestParameterMutability(unittest.TestCase):
    """Test that Flow.parameters always returns references
    of underlying flow parameters, so that modifications of those
    parameters update the actual parameters in the relevant Flow
    instance."""

    # Flows with intentionally static parameters
    EXCLUDE = ["Permutation"]

    def test_parameter_mutability(self):
        d = 4
        for flow_name, flow_cls in FLOWS.items():
            if flow_name in self.EXCLUDE:
                continue
            flow = flow_cls(d)
            pars = flow.parameters
            for k, v in pars.items():
                assert len(v.shape) == 1  # current limitation
                for i in range(v.shape[0]):
                    v[i] += 10.0
                    assert abs(getattr(flow, k)[i] - v[i]) < 1e-6


class TestJacDet(unittest.TestCase):
    """Test accuracy of Jacobian determinant calculation for
    all Flows."""

    def test_jac_det(self):
        d = 4
        N = 10
        delta = 1e-4
        X = np.random.normal(size=(N, d)).astype(DTYPE)
        for flow_name, flow_cls in FLOWS.items():
            flow = flow_cls(d)

            # Evaluate Jacobian determinants analytically
            Y, jacdetana = flow.forward(X)

            # Evaluate Jacobian determinants numerically
            def f(x):
                X = np.array([x])
                Y, _ = flow.forward(X)
                return Y[0, :]

            jacdetnum = np.zeros(N, dtype=DTYPE)
            for i in range(N):
                jacdetnum[i] = numerical_jacdet(f, X[i, :], delta=delta)

            np.testing.assert_allclose(jacdetana, jacdetnum, atol=1e-5, rtol=1e-5)


class TestInversion(unittest.TestCase):
    """Test equivalence of forward and inverse calculation for
    all Flows."""

    def setUp(self):
        np.random.seed(666)

    def test_inversion(self):
        d = 4
        N = 10
        delta = 1e-4
        X = np.random.normal(size=(N, d)).astype(DTYPE)
        for flow_name, flow_cls in FLOWS.items():
            flow = flow_cls(d)
            Y, _ = flow.forward(X)
            Z = flow.invert(Y)
            np.testing.assert_allclose(Z, X, atol=1e-5, rtol=1e-5)


class TestGradient(unittest.TestCase):
    """Test that we can compute gradients of model parameters
    for all Flows. We use the tools from the Model class for
    this test."""

    def setUp(self):
        np.random.seed(666)

    def test_gradient(self):
        d = 4
        N = 10
        delta = 1e-4
        X = np.random.normal(size=(N, d)).astype(DTYPE)
        for flow_name, flow_cls in FLOWS.items():
            flow = flow_cls(d)
            model = Model([flow])
            _, _, dL_dpars_ana = model.backward(X)

            def loss(params: np.ndarray) -> float:
                model.set_parameters(params)
                Y, dJ = flow.forward(X)
                L = 0.5 * (Y**2).sum(axis=1) + (d / 2) * np.log(2 * np.pi)
                L -= np.log(np.abs(dJ) + EPSILON)
                return L.mean()

            params = model.get_parameters(flat=True)
            dL_dpars_num = finite_differences(loss, params, delta=delta)
            np.testing.assert_allclose(dL_dpars_ana, dL_dpars_num, atol=1e-3, rtol=1e-3)
