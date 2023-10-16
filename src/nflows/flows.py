"""Implementations of flow layers."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple
import numpy as np
from .constants import DTYPE, EPSILON


class Flow(ABC):
    """Represents an invertible operation Y = f(X)."""

    @classmethod
    @abstractmethod
    def from_shape(cls, shape: tuple, params: np.ndarray = None):
        """Make an instance of this Flow class from a shape and
        a linear array of parameters. Useful for deserialization.
        If *params* is not passed, should instantiate with suitable
        defaults."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        """Shape of the inputs and outputs to this Flow.
        First dimension is assumed to encode index in
        batch (e.g. None)."""

    @property
    @abstractmethod
    def parameters(self) -> OrderedDict:
        """Get a dict of all model parameters, keyed by
        parameter name."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        """Transform samples of random variable X into
        samples of random variable Y.

        Parameters
        ----------
        X   :   input to this Flow; first dimension is assumed
                to encode different samples

        Returns
        -------
        0:  Y, output of this flow
        1:  Jacobian determinant for each sample; ndarray of
            shape (X.shape[0],)
        """

    @abstractmethod
    def invert(self, Y: np.ndarray) -> np.ndarray:
        """Transform samples of random variable Y into
        samples of random variable X."""

    @abstractmethod
    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        """Given partial derivatives of a scalar loss with
        respect to the output of this Flow, propagate these
        partial derivatives to the Flow's input and its
        parameters.

        Parameters
        ---------
        X           :   input to this Flow
        dL_dY       :   partial derivatives of a scalar loss
                        with respect to each output element
                        of this loss (shape *X.shape*)
        normalize   :   actually normalize the propagated
                        partial derivatives using the Jac.
                        determinant. If False, we only
                        backpropagate through the output of
                        the Flow, not through its determinant.
                        Useful for tests.

        Returns
        -------
        0:  partial derivatives of loss with respect to each
            element in input X (shape *X.shape*);
        1:  partial derivatives of log absolute Jacobian
            determinant with respect to each element in the
            input (shape *X.shape*);
        2:  dict keyed by parameter name, partial derivatives
            of loss with respect to each parameter (gradient)
        """


class AffineFlow(Flow):
    def __init__(self, d: int, mean=None, scale=None):
        self.d = d
        if mean is None:
            mean = np.zeros(d, dtype=DTYPE)
        if scale is None:
            scale = np.ones(d, dtype=DTYPE)
        assert len(mean.shape) == len(scale.shape) == 1
        assert mean.shape == scale.shape == (d,)
        self.mean = mean
        self.scale = scale

    @classmethod
    def from_shape(cls, shape: tuple, params: np.ndarray = None):
        assert isinstance(shape, tuple)
        assert len(shape) == 1
        d = shape[0]
        mean = None
        scale = None
        if params is not None:
            assert isinstance(params, np.ndarray)
            assert len(params.shape) == 1
            assert params.shape == (2 * d,)
            mean = params[:d]
            scale = params[d:]
        return cls(shape[0], mean=mean, scale=scale)

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict(mean=self.mean, scale=self.scale)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        Y = X * self.scale + self.mean
        detjac = np.full(X.shape[0], np.prod(self.scale), dtype=X.dtype)
        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / (self.scale + np.sign(self.scale) * EPSILON)

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        a = self.scale
        b = self.mean
        dL_dX = dL_dY * a
        dlogdetjac_dX = np.zeros_like(X)
        # if normalize:  # meaningless for AffineFlow
        #     dL_dX -= dlogdetjac_dX
        dL_dpars = {}
        dL_dpars["mean"] = dL_dY.mean(axis=0)
        dL_dpars["scale"] = (dL_dY * X).mean(axis=0) - 1 / (a + np.sign(a) * EPSILON)
        return dL_dX, dlogdetjac_dX, dL_dpars


class PlanarFlow(Flow):
    def __init__(self, d: int, w=None, v=None, b=None):
        self.d = d
        if w is None:
            w = np.random.normal(scale=np.sqrt(1 / d), size=d).astype(DTYPE)
        if v is None:
            v = np.random.normal(scale=np.sqrt(1 / d), size=d).astype(DTYPE)
        if b is None:
            b = np.array([0.0])
        assert len(w.shape) == len(v.shape) == len(b.shape) == 1
        assert w.shape == v.shape == (d,)
        assert b.shape == (1,)
        self.w = w
        self.v = v
        self.b = b

    @classmethod
    def from_shape(cls, shape: tuple, params: np.ndarray = None):
        assert isinstance(shape, tuple)
        assert len(shape) == 1
        d = shape[0]
        w = None
        v = None
        b = None
        if params is not None:
            assert isinstance(params, np.ndarray)
            assert len(params.shape) == 1
            assert params.shape == (2 * d + 1,)
            w = params[:d]
            v = params[d : 2 * d]
            b = params[2 * d :]
        return cls(d, w=w, v=v, b=b)

    @property
    def shape(self) -> Tuple[int]:
        return (None, self.d)

    @property
    def parameters(self) -> OrderedDict:
        return OrderedDict(w=self.w, v=self.v, b=self.b)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b
        tanha = np.tanh(X @ w + b)
        Y = X + tanha[:, None] * v
        wv = w @ v
        dtanha_da = 1 - tanha**2
        detjac = 1 + wv * dtanha_da
        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        assert len(Y.shape) == 2
        assert Y.shape[1] == self.d
        n = Y.shape[0]
        alpha = np.zeros(n, dtype=Y.dtype)
        converged = np.zeros(n, dtype=bool)
        w = self.w
        v = self.v
        b = self.b
        wv = w @ v
        assert wv > -1  # invertibility criterion
        wy = Y @ w
        while not converged.all():
            sigma = np.tanh(alpha + b)
            dsigma_dalpha = 1.0 - sigma**2
            dg_dalpha = wv * dsigma_dalpha + 1.0
            g = alpha + wv * sigma - wy
            update = -g / (2 * dg_dalpha)
            converged = np.abs(update) < 1e-6
            alpha += update
        X = Y - np.tanh(alpha + b)[:, None] * v
        return X

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        assert len(X.shape) == len(dL_dY.shape) == 2
        assert X.shape[1] == dL_dY.shape[1] == self.d
        w = self.w
        v = self.v
        b = self.b

        # Gradient of loss with respect to inputs to this Flow
        a = X @ w + b
        tanha = np.tanh(a)
        dtanha_da = 1 - tanha**2
        ddtanha_dda = -2 * tanha * dtanha_da
        dyv = dL_dY @ v
        dL_dX = dL_dY + (dtanha_da * dyv)[:, None] * w

        # Jacobian determinant
        wv = w @ v
        detjac = 1 + wv * dtanha_da

        # Gradient of loss with respect to log Jacobian determinant
        dlogdetjac_dX = ((wv * ddtanha_dda) / (detjac + EPSILON))[:, None] * w
        if normalize:
            dL_dX -= dlogdetjac_dX

        # Gradient of loss with respect to all parameters of this Flow
        dL_dpars = {}
        invdetjac = 1 / (detjac + np.sign(detjac) * EPSILON)
        dL_dpars["w"] = (
            ((dL_dY @ v) * dtanha_da)[:, None] * X
            - (
                (dtanha_da * invdetjac)[:, None] * v
                + wv * (ddtanha_dda * invdetjac)[:, None] * X
            )
        ).mean(axis=0)
        dL_dpars["v"] = (
            tanha[:, None] * dL_dY - (dtanha_da * invdetjac)[:, None] * w
        ).mean(axis=0)
        dL_dpars["b"] = ((dL_dY @ v) * dtanha_da - wv * ddtanha_dda * invdetjac)[
            :, None
        ].mean(axis=0)

        return dL_dX, dlogdetjac_dX, dL_dpars


class Permutation(Flow):
    def __init__(self, d: int, order: np.ndarray = None):
        self.d = d
        if order is None:
            order = np.arange(d)
            while (order == np.arange(d)).all():
                np.random.shuffle(order)
        self.true_order = np.round(order, 0).astype(np.int32)
        self.order = self.true_order.astype(np.float32)

        # Because this layer does not train, precompute
        # the Jacobian determinant. This is always 1 or -1.
        P = np.zeros((d, d), dtype=DTYPE)
        P[np.arange(d), self.true_order] = 1.0
        self.detjac = np.linalg.det(P)

    @classmethod
    def from_shape(cls, shape: tuple, params: np.ndarray = None):
        assert isinstance(shape, tuple)
        assert len(shape) == 1
        d = shape[0]
        order = None
        if params is not None:
            assert isinstance(params, np.ndarray)
            assert params.shape[0] == (d,)
            order = np.round(params, 0).astype(np.int32)
        return cls(d, order=order)

    @property
    def shape(self) -> Tuple[int]:
        return (self.d,)

    @property
    def parameters(self) -> OrderedDict:
        return {"order": self.true_order.astype(np.float32)}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        order = self.true_order
        Y = X[:, order]
        detjac = np.full(X.shape[0], self.detjac, dtype=DTYPE)
        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        order = self.true_order
        inv_order = np.argsort(order)
        X = Y[:, inv_order]
        return X

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        assert len(X.shape) == 2
        assert len(dL_dY.shape) == 2
        assert X.shape[1] == self.d
        assert dL_dY.shape[1] == self.d
        assert X.shape == dL_dY.shape
        order = self.true_order
        inv_order = np.argsort(order)
        dL_dX = dL_dY[:, inv_order]
        dlogdetjac_dX = np.zeros_like(X)
        # if normalize:  # meaningless for Permutation
        #     dL_dX -= dlogdetjac_dX
        dL_dpars = {"order": np.zeros(self.d, dtype=DTYPE)}
        return dL_dX, dlogdetjac_dX, dL_dpars


class RadialFlow(Flow):
    def __init__(
        self,
        d: int,
        b: np.ndarray = None,
        alpha: np.ndarray = None,
        beta: np.ndarray = None,
    ):
        if b is None:
            b = np.zeros(d, dtype=DTYPE)
        if alpha is None:
            alpha = np.array([1.0])
        if beta is None:
            beta = np.array([0.0])

        assert b.shape == (d,)
        assert alpha.shape == (1,)
        assert beta.shape == (1,)

        self.d = d
        self.b = b
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def from_shape(cls, shape: Tuple[int], params: np.ndarray = None):
        assert len(shape) == 1
        d = shape[0]
        b = None
        alpha = None
        beta = None
        if params is not None:
            assert params.shape == (d + 2,)
            b = params[:d]
            alpha = params[d : d + 1]
            beta = params[d + 1 :]
        return cls(d, b=b, alpha=alpha, beta=beta)

    @property
    def shape(self) -> tuple:
        return (None, self.d)

    @property
    def parameters(self) -> dict:
        return {"b": self.b, "alpha": self.alpha, "beta": self.beta}

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray]:
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        dX = X - b
        R = np.sqrt((dX**2).sum(axis=1))
        alphaplus = np.log(1.0 + np.exp(alpha) + EPSILON)
        Ralpha = R + alphaplus
        Y = X + beta * dX / (Ralpha + EPSILON)[:, None]

        # mag1 = beta / (Ralpha + EPSILON)
        # mag3 = beta / (R * Ralpha**2 + EPSILON)
        # detjac = 1 + mag1 - mag3 * (dX**2).sum(axis=1)

        mag1 = 1 + beta / (Ralpha + EPSILON)
        mag3 = beta / (R * Ralpha**2 + EPSILON)
        detjac = (mag1 ** (self.d - 1)) * (mag1 + mag3 * R)

        return Y, detjac

    def invert(self, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(
        self, X: np.ndarray, dL_dY: np.ndarray, normalize: bool = True
    ) -> tuple:
        assert X.shape == dL_dY.shape
        assert len(X.shape) == 2
        assert X.shape[1] == self.d
        b = self.b
        alpha = self.alpha
        beta = self.beta
        alphaplus = np.log(1.0 + np.exp(alpha) + EPSILON)
        dX = X - b
        R = np.sqrt((dX**2).sum(axis=1))
        Ralpha = R + alphaplus
        dL_dX = (1 + beta / (Ralpha + EPSILON))[:, None] * dL_dY
        dL_dX -= (
            beta
            * (dL_dY * dX).sum(axis=1)[:, None]
            * dX
            / (R * (Ralpha**2) + EPSILON)[:, None]
        )
        dlogdetjac_dX = None
        dL_dpars = {}
        mag1 = beta / (Ralpha + EPSILON)
        mag2 = beta / (Ralpha + EPSILON) ** 2
        mag3 = beta / (R * Ralpha**2 + EPSILON)
        dL_dpars["alpha"] = (
            -((dL_dY * dX).sum(axis=1) * mag2).mean() * 1.0 / (1.0 + np.exp(-alpha))
        )
        dL_dpars["beta"] = ((dL_dY * dX).sum(axis=1) / (Ralpha + EPSILON)).mean()
        dL_dpars["b"] = (
            -mag1[:, None] * dL_dY + (mag3 * (dL_dY * dX).sum(axis=1))[:, None] * dX
        ).mean(axis=0)
        # raise NotImplementedError # still need to factor in log Jacobian
        return dL_dX, dlogdetjac_dX, dL_dpars


FLOWS = {f.__name__: f for f in [AffineFlow, PlanarFlow, Permutation]}

FLOW_HASHES = {
    1: "AffineFlow",
    2: "PlanarFlow",
    3: "Permutation",
}
