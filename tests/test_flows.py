import numpy as np
import unittest

from nflows.constants import DTYPE
from nflows.flows import LinearFlow, PlanarFlow
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

    def test_get_parameters(self):
        flow = PlanarFlow(self.d)
        pars = flow.get_parameters()
        assert pars["w"] is flow.w
        assert pars["v"] is flow.v
        assert abs(pars["b"] - flow.b) < 1e-6
        # numpy arrays are references
        pars["w"][0] += 1.0
        assert abs(pars["w"][0] - flow.w[0]) < 1e-6

    def test_logdetjac(self):
        """Makes sure that the method PlanarFlow.dlogdetjac_dX
        truly gives the derivative of Planar.logdetjac with
        respect to its input."""
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
        delta = 1e-3

        for X in sample_X:

            def f(X: np.ndarray) -> float:
                return flow.logdetjac(X)[0]

            dlogdetjac_dX_num = finite_differences(f, X, delta=delta)
            dlogdetjac_dX = flow.dlogdetjac_dX(X)
            np.testing.assert_allclose(
                dlogdetjac_dX, dlogdetjac_dX_num, atol=1e-4, rtol=1e-3
            )

    def test_backward(self):
        w = np.random.normal(size=self.d).astype(DTYPE)
        v = np.random.normal(size=self.d).astype(DTYPE)
        b = np.random.normal()
        flow = PlanarFlow(n=self.d, w=w, v=v, b=b)

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
            dL_dX_ana = flow.backward(Xi, dL_dY)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)


class TestLinearFlow(unittest.TestCase):
    def setUp(self):
        self.n = 4
        np.random.seed(666)

    def test_get_parameter(self):
        mean = np.random.normal(size=self.n).astype(DTYPE)
        scale = np.random.normal(size=self.n).astype(DTYPE)
        flow = LinearFlow(n=self.n, mean=mean, scale=scale)
        pars = flow.get_parameters()
        assert pars["mean"] is flow.mean
        assert pars["scale"] is flow.scale
        np.testing.assert_allclose(pars["mean"], flow.mean, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(pars["scale"], flow.scale, atol=1e-5, rtol=1e-5)

    def test_invert(self):
        mean = np.random.normal(size=self.n).astype(DTYPE)
        scale = np.random.normal(size=self.n).astype(DTYPE)
        flow = LinearFlow(n=self.n, mean=mean, scale=scale)
        X = np.random.normal(size=(10, self.n)).astype(DTYPE)
        Y = flow.forward(X)
        Z = flow.invert(Y)
        np.testing.assert_allclose(Z, X, atol=1e-4, rtol=1e-4)

    def test_backward(self):
        mean = np.random.normal(size=self.n).astype(DTYPE)
        scale = np.random.normal(size=self.n).astype(DTYPE)
        flow = LinearFlow(n=self.n, mean=mean, scale=scale)

        def loss(X: np.ndarray) -> float:
            Y = flow.forward(X)
            return 0.5 * (Y**2).sum()

        X = np.random.normal(size=(10, self.n)).astype(DTYPE)

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
            dL_dX_ana = flow.backward(Xi, dL_dY)
            np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-3, rtol=1e-3)
