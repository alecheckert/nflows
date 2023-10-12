import numpy as np
from typing import Callable


def finite_differences(f: Callable, X: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """Evaluate the numerical derivative of a scalar-valued function
    f with respect to each of its inputs via the finite differences
    method. Useful for validating backpropagation procedures.

    Parameters
    ----------
    f       :   function with signature f(X: np.ndarray) -> float
    X       :   ndarray, inputs
    delta   :   size of finite difference to use

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
    return df_dX.reshape(X.shape)
