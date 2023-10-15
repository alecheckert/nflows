import numpy as np
from typing import Callable


def finite_differences(f: Callable, X: np.ndarray, delta: float = 1e-3, verbose: bool = False) -> np.ndarray:
    """Evaluate the numerical derivative of a scalar-valued function
    f with respect to each of its inputs via the finite differences
    method. Useful for validating backpropagation procedures.

    Parameters
    ----------
    f       :   function with signature f(X: np.ndarray) -> float
    X       :   ndarray, inputs
    delta   :   size of finite difference to use
    verbose :   print intermediate values

    Returns
    -------
    ndarray of same shape as X, derivatives of f with respect to
    each element of X
    """
    assert isinstance(X, np.ndarray)
    Xflat = X.ravel()
    n = Xflat.shape[0]
    hd = delta / 2.0
    df_dX = np.zeros_like(Xflat)
    for i in range(n):
        base = Xflat[i]
        Xflat[i] = base - hd
        f0 = f(Xflat.reshape(X.shape))
        Xflat[i] = base + hd
        f1 = f(Xflat.reshape(X.shape))
        df_dX[i] = (f1 - f0) / delta
        if verbose:
            print(f"\nf(X[{i}]-{hd}) =\t{f0}")
            print(f"f(X[{i}]+{hd}) =\t{f1}")
            print(f"df/dX[{i}]   ~=\t{df_dX[i]}")
        Xflat[i] = base
    return df_dX.reshape(X.shape)


def numerical_jacdet(f: Callable, x: np.ndarray, delta: float = 1e-3) -> float:
    """Estimate the Jacobian determinant of a single data point.

    Parameters
    ----------
    f       :   function with signature f(x: np.ndarray) -> np.ndarray,
                where input and output are 1D arrays of same size
    x       :   input (1D array)
    delta   :   size of finite difference to use

    Returns
    -------
    float, estimated Jacobian determinant
    """
    assert len(x.shape) == 1
    n = x.shape[0]
    hd = 0.5 * delta
    J = np.zeros((n, n), dtype=x.dtype)
    for i in range(n):
        base = x[i]
        x[i] = base - hd
        y0 = f(x)
        x[i] = base + hd
        y1 = f(x)
        J[:,i] = (y1 - y0) / delta
        x[i] = base
    return np.linalg.det(J)
