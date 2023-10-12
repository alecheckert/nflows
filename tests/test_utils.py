import numpy as np
import unittest
from nflows.utils import finite_differences


class TestFiniteDifferences(unittest.TestCase):
    def test_finite_differences_linear(self):
        slope = 3.0
        offset = -2.0
        delta = 1e-3

        def f(X: np.ndarray) -> float:
            return (X * slope + offset).mean()

        X = np.random.normal(size=(10, 3, 5))
        size = np.prod(X.shape)
        df_dX = finite_differences(f, X, delta=delta)
        assert df_dX.shape == X.shape
        np.testing.assert_allclose(df_dX, slope / size, atol=1e-5, rtol=1e-5)

    def test_finite_differences_quadratic(self):
        delta = 1e-3

        def f(X: np.ndarray) -> float:
            return (X**2).mean()

        X = np.random.normal(size=(10, 3, 5))
        size = np.prod(X.shape)
        # Numerical derivatives
        df_dX_num = finite_differences(f, X, delta=delta)
        # Analytical derivatives
        df_dX_ana = (2 / size) * X
        np.testing.assert_allclose(df_dX_num, df_dX_ana, atol=1e-5, rtol=1e-5)
