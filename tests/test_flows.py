import numpy as np
import unittest

from nflows.constants import DTYPE, EPSILON
from nflows.flows import LinearFlow, PlanarFlow, FLOWS
from nflows.utils import finite_differences


class TestPlanarFlow(unittest.TestCase):
    def setUp(self):
        self.d = 4
        self.N = 10
        np.random.seed(666)
        self.X = np.random.normal(size=(self.N, self.d)).astype(DTYPE)

    def test_forward(self):
        """Does-it-run? type test"""
        flow = PlanarFlow(self.d)
        Y = flow.forward(self.X)
        assert Y.shape == (self.N, self.d)

    def test_invert(self):
        flow = PlanarFlow(self.d)
        X = self.X
        Y = flow.forward(X)
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
                _, detjac, _ = flow.backward(X, np.zeros_like(X))
                return np.log(detjac[0] + EPSILON)

            _, detjac, dlogdetjac_dX = flow.backward(X, np.zeros_like(X))
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
            Y = flow.forward(X)
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
            Y = flow.forward(Xi)
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
            Y = flow.forward(X)
            L = 0.5 * (Y**2).sum() + (self.d / 2) * np.log(2 * np.pi)
            _, detjac, _ = flow.backward(X, np.zeros_like(X))
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
            Y = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)


class TestLinearFlow(unittest.TestCase):
    def setUp(self):
        self.d = 4
        np.random.seed(666)

    def test_parameters(self):
        mean = np.random.normal(size=self.d).astype(DTYPE)
        scale = np.random.normal(size=self.d).astype(DTYPE)
        flow = LinearFlow(d=self.d, mean=mean, scale=scale)
        pars = flow.parameters
        assert pars["mean"] is flow.mean
        assert pars["scale"] is flow.scale
        np.testing.assert_allclose(pars["mean"], flow.mean, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(pars["scale"], flow.scale, atol=1e-5, rtol=1e-5)

    def test_invert(self):
        mean = np.random.normal(size=self.d).astype(DTYPE)
        scale = np.random.normal(size=self.d).astype(DTYPE)
        flow = LinearFlow(d=self.d, mean=mean, scale=scale)
        X = np.random.normal(size=(10, self.d)).astype(DTYPE)
        Y = flow.forward(X)
        Z = flow.invert(Y)
        np.testing.assert_allclose(Z, X, atol=1e-4, rtol=1e-4)

    def test_backward(self):
        mean = np.random.normal(size=self.d).astype(DTYPE)
        scale = np.random.normal(size=self.d).astype(DTYPE)
        flow = LinearFlow(d=self.d, mean=mean, scale=scale)

        def loss(X: np.ndarray) -> float:
            Y = flow.forward(X)
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
            Y = flow.forward(Xi)
            dL_dY = Y.copy()
            dL_dX_ana, _, _ = flow.backward(Xi, dL_dY, normalize=False)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)

    def test_jacdet(self):
        mean = np.random.normal(size=self.d).astype(DTYPE)
        scale = np.random.normal(size=self.d).astype(DTYPE)
        flow = LinearFlow(d=self.d, mean=mean, scale=scale)
        n = 3
        X = np.random.normal(size=(1, self.d)).astype(DTYPE)
        _, jd, dlogdetjac_dX = flow.backward(X, np.zeros_like(X))
        assert jd.shape == (X.shape[0],)
        np.testing.assert_allclose(jd, np.prod(scale), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(dlogdetjac_dX, 0.0, atol=1e-4, rtol=1e-4)


class TestParameterMutability(unittest.TestCase):
    """Test that Flow.parameters always returns references
    of underlying flow parameters, so that modifications of those
    parameters update the actual parameters in the relevant Flow
    instance."""

    def test_parameter_mutability(self):
        d = 4
        for flow_name, flow_cls in FLOWS.items():
            flow = flow_cls(d)
            pars = flow.parameters
            for k, v in pars.items():
                assert len(v.shape) == 1  # current limitation
                for i in range(v.shape[0]):
                    v[i] += 10.0
                    assert abs(getattr(flow, k)[i] - v[i]) < 1e-6
