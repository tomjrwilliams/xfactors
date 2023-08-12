
from __future__ import annotations

import operator
import collections
# import collections.abc
import functools
import itertools

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import jaxopt
import optax

import xtuples as xt
from ... import xfactors as xf

from . import funcs

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Constraint_Orthonormal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_Orthonormal, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return funcs.loss_orthonormal(X)


@xt.nTuple.decorate()
class Constraint_Orthogonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_Orthogonal, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return funcs.loss_orthogonal(X)


@xt.nTuple.decorate()
class Constraint_VOrthogonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_VOrthogonal, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return jax.vmap(funcs.loss_orthogonal)(X).sum()


@xt.nTuple.decorate()
class Constraint_VOrthonormal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_VOrthonormal, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return jax.vmap(funcs.loss_orthonormal)(X).sum()


@xt.nTuple.decorate()
class Constraint_VDiagonal(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_VDiagonal, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        X = self.data.access(state)
        if self.T:
            X = X.T
        return jax.vmap(funcs.loss_diag)(X).sum()


@xt.nTuple.decorate()
class Constraint_XXt_Cov(typing.NamedTuple):
    
    data: xf.Location
    cov: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_XXt_Cov, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        
        X = self.data.access(state)
        cov = self.cov.access(state)

        XXt = jax.numpy.matmul(X, X.T)

        return funcs.loss_mse(XXt, cov)


@xt.nTuple.decorate()
class Constraint_EigenVLike(typing.NamedTuple):
    
    weights: xf.Location
    factors: xf.Location

    eigval_max: bool = True

    n_check: typing.Optional[int] = None

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_EigenVLike, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        w = self.weights.access(state)
        f = self.factors.access(state)

        cov = jax.numpy.cov(f.T)
        eigvals = jax.numpy.diag(cov)

        if self.n_check is not None:
            assert eigvals.shape[0] == self.n_check, (
                self, eigvals.shape,
            )
        res = (
            + funcs.loss_descending(eigvals)
            + funcs.loss_orthogonal(w.T)
            + funcs.loss_mean_zero(0)(f)
            + funcs.loss_diag(cov)
        )
        if self.eigval_max:
            return res + (
                - jax.numpy.sum(jax.numpy.log(1 + eigvals))
                # ridge penalty to counteract eigval max
            )
        return res
    
def l1_diag_loss(v):
    return jax.numpy.abs(jax.numpy.diag(v)).mean()

@xt.nTuple.decorate()
class Constraint_L1_MM_Diag(typing.NamedTuple):
    
    raw: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_L1_MM_Diag, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        raw = self.raw.access(state)
        data = jax.numpy.matmul(
            jax.numpy.transpose(raw, (0, 2, 1)),
            raw,
        )
        return jax.vmap(l1_diag_loss)(data).mean()

# ---------------------------------------------------------------



@xt.nTuple.decorate()
class Constraint_KernelVsCov(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_KernelVsCov, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # param weights
        # y_pred = weights . normal
        # mse(y_pred, y)
        # 
        # cov_weights = weights.weights
        # cov_kernel = kernel(latent(x))
        # mse(cov_weights, cov_kernel)
        assert False, self


# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Constraint_MinimiseMMSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MinimiseMMSpread, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        cov = jax.numpy.matmul(
            jax.numpy.transpose(data, (0, 2, 1)),
            data,
        )
        mu = jax.numpy.mean(cov, axis = 0)
        delta = jax.numpy.square(jax.numpy.subtract(
            cov,
            xf.expand_dims(mu, 0, cov.shape[0])
        )).mean()
        return delta



@xt.nTuple.decorate()
class Constraint_MinimiseVariance(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MinimiseVariance, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        var = jax.numpy.var(data.flatten())
        return var


@xt.nTuple.decorate()
class Constraint_MinimiseZSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MinimiseZSpread, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        var = jax.numpy.var(data.flatten())
        sigma = jax.numpy.sqrt(var)
        mu = jax.numpy.mean(data)
        delta = (data - mu) / sigma
        return jax.numpy.square(delta).mean()


@xt.nTuple.decorate()
class Constraint_MaxSpread(typing.NamedTuple):
    
    data: xf.Location
    T: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MaxSpread, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        if self.T:
            data = data.T
        data = xf.expand_dims(data, 0, data.shape[0])
        dataT = jax.numpy.transpose(
            data, (1, 0, 2,)
        )
        delta = jax.numpy.abs(
            jax.numpy.subtract(data, dataT)
        ).sum(axis=-1)
        return -1 * delta.mean()

# ---------------------------------------------------------------

