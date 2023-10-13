import numpy as np
import unittest
from nflows.utils import finite_differences, numerical_jacdet


class TestFiniteDifferences(unittest.TestCase):
    def test_linear(self):
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

    def test_quadratic(self):
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

    def test_harmonic(self):
        delta = 1e-4
        a = 0.5
        b = -2.5
        c = 5.0

        def f(X: np.ndarray) -> float:
            return (b * np.cos(a * X) + c).sum()

        X = np.random.normal(size=(10, 3, 5))
        size = np.prod(X.shape)
        # Numerical derivatives
        df_dX_num = finite_differences(f, X, delta=delta)
        # Analytical derivatives
        df_dX_ana = -a * b * np.sin(a * X)
        np.testing.assert_allclose(df_dX_num, df_dX_ana, atol=1e-3, rtol=1e-2)


class TestNumericalJacDet(unittest.TestCase):
    def test_scalar(self):
        d = 4
        delta = 1e-4
        a = np.random.normal(scale=2, size=d)
        b = np.random.normal(scale=2, size=d)
        x = np.random.normal(scale=2, size=d)

        def f(x: np.ndarray) -> np.ndarray:
            return a * x + b

        detjac = numerical_jacdet(f, x, delta=delta)
        assert abs(detjac - np.prod(a)) < 1e-6
