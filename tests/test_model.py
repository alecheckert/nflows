import numpy as np
import unittest

from nflows.constants import DTYPE, EPSILON
from nflows.flows import PlanarFlow, ScalarFlow
from nflows.model import Model
from nflows.utils import finite_differences, numerical_jacdet


class TestModelScalar1(unittest.TestCase):
    """Simple Model comprised of a single ScalarFlow."""

    def setUp(self):
        np.random.seed(666)
        self.d = 4
        self.N = 10
        self.mean = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        self.scale = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        self.X = np.random.normal(size=(self.N, self.d)).astype(DTYPE)
        self.flow = ScalarFlow(self.d, mean=self.mean, scale=self.scale)
        self.model = Model(flows=[self.flow])

    def test_init(self):
        model = self.model
        assert len(model.flows) == 1
        assert model.flows[0] is self.flow
        for par in ["mean", "scale"]:
            np.testing.assert_allclose(
                model.flows[0].parameters.get(par),
                self.flow.parameters[par],
                atol=1e-6,
                rtol=1e-6,
            )

    def test_get_set_parameters(self):
        model = self.model
        params = model.get_parameters(flat=True)
        assert params.shape == (self.d * 2,)
        np.testing.assert_allclose(params[: self.d], self.mean, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(params[self.d :], self.scale, atol=1e-6, rtol=1e-6)

        # Test Model.set_parameters
        params = params * 2 + 3.0
        model.set_parameters(params.copy())
        params2 = model.get_parameters(flat=True)
        np.testing.assert_allclose(params, params2, atol=1e-6, rtol=1e-6)

        # Test Model.split_parameters
        params_split = model.split_parameters(params2)
        for (flow_idx, parname), p in params_split.items():
            np.testing.assert_allclose(
                model.flows[flow_idx].parameters[parname], p, atol=1e-6, rtol=1e-6
            )

    def test_forward(self):
        model = self.model
        X = self.X
        Y0 = model.forward(X)
        Y1, _ = self.flow.forward(X)
        np.testing.assert_allclose(Y0, Y1, atol=1e-6, rtol=1e-6)

    def test_invert(self):
        model = self.model
        X = self.X
        X0 = self.flow.invert(self.flow.forward(X)[0])
        X1 = model.invert(model.forward(X))
        np.testing.assert_allclose(X0, X1, atol=1e-6, rtol=1e-6)

    def test_generate(self):
        model = self.model
        n = 8
        Xsample = model.generate(n)
        assert Xsample.shape == (n, self.d)
        assert np.isfinite(Xsample).all()

    def test_backward(self):
        model = self.model
        flow = self.flow
        X = self.X.copy()
        delta = 1e-4
        loss, dL_dX_ana, dL_dpars_ana = model.backward(X)

        # Does not modify input
        np.testing.assert_allclose(X, self.X, atol=1e-6, rtol=1e-6)

        # Check accuracy of loss value
        Y, dJ = flow.forward(X)
        loss_expected = (
            0.5 * (Y**2).sum(axis=1)
            + (self.d / 2) * np.log(2 * np.pi)
            - np.log(np.abs(dJ) + EPSILON)
        )
        np.testing.assert_allclose(loss, loss_expected, atol=1e-5, rtol=1e-5)

        # Test for numerical accuracy of derivatives of loss w.r.t. input
        def f(X: np.ndarray) -> float:
            L, _, _ = model.backward(X)
            return L.sum()

        dL_dX_num = finite_differences(f, X, delta=delta)
        np.testing.assert_allclose(dL_dX_ana, dL_dX_num, atol=1e-5, rtol=1e-5)

        # Test for numerical accuracy of derivatives of loss w.r.t. parameters
        def f(params: np.ndarray) -> float:
            model.set_parameters(params)
            L, _, _ = model.backward(X)
            return L.mean()

        params = model.get_parameters(flat=True)
        dL_dpars_num = finite_differences(f, params, delta=delta)
        np.testing.assert_allclose(dL_dpars_ana, dL_dpars_num, atol=1e-4, rtol=1e-3)


class TestModelPlanar(unittest.TestCase):
    """More complex Model comprised of a stacked ScalarFlow and PlanarFlow."""

    def setUp(self):
        np.random.seed(666)
        self.d = 4
        self.N = 10
        mean = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        scale = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        w = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        v = np.random.normal(scale=2, size=self.d).astype(DTYPE)
        b = np.random.normal(scale=2, size=1).astype(DTYPE)
        assert w @ v > -1  # condition for model invertibility
        self.X = np.random.normal(size=(self.N, self.d)).astype(DTYPE)
        self.model = Model(
            flows=[
                ScalarFlow(self.d, mean=mean, scale=scale),
                PlanarFlow(self.d, w=w, v=v, b=b),
            ]
        )

    def test_init(self):
        model = self.model
        assert len(model.flows) == 2
        assert isinstance(model.flows[0], ScalarFlow)
        assert isinstance(model.flows[1], PlanarFlow)

    def test_get_parameters(self):
        model = self.model
        params = model.get_parameters(flat=False)
        flat_params = model.get_parameters(flat=True)
        for (i, k), v in params.items():
            assert params[(i, k)] is model.flows[i].parameters[k]
            s0, s1 = model.parameter_map[(i, k)]
            np.testing.assert_allclose(
                v, model.flows[i].parameters[k], atol=1e-6, rtol=1e-6
            )
            np.testing.assert_allclose(v, flat_params[s0:s1], atol=1e-6, rtol=1e-6)
            # Can modify underlying parameters by reference
            v += np.ones(v.shape, dtype=DTYPE)
            np.testing.assert_allclose(
                v, model.flows[i].parameters[k], atol=1e-6, rtol=1e-6
            )
