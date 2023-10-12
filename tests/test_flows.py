import numpy as np
import unittest

from nflows.flows import PlanarFlow, DTYPE


class TestPlanarFlow(unittest.TestCase):
    def setUp(self):
        np.random.seed(666)
        self.d = 4
        self.N = 10
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
