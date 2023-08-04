
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

from . import rand
from . import dates
from . import xfactors as xf

# ---------------------------------------------------------------

def loss_mse(l, r):
    return jax.numpy.square(jax.numpy.subtract(l, r)).mean()

def loss_mse_zero(X1):
    return jax.numpy.square(X1).mean()

@functools.lru_cache(maxsize=4)
def loss_mean_zero(axis):
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    return f

# ascending just reverse order of xl and xr
def loss_descending(x):
    order = jax.numpy.flip(jax.numpy.argsort(x))
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()

def loss_diag(X, diag):
    diag = jax.numpy.multiply(
        jax.numpy.eye(X.shape[0]), diag
    )
    return loss_mse(X, diag)

def loss_orthonormal(X, scale = 1.):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T) / scale
    return loss_mse(XXt, eye)

def loss_orthogonal(X, scale = 1.):
    XXt = jax.numpy.matmul(X, X.T) / scale
    return loss_diag(
        XXt, jax.numpy.diag(XXt)
    )

# ---------------------------------------------------------------

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_Maximise(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        vals = xf.concatenate_sites(self.sites, state)
        return -1 * vals.mean()

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_Minimise(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        vals = xf.concatenate_sites(self.sites, state)
        return vals.mean()

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_MinimiseSquare(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        vals = xf.concatenate_sites(self.sites, state)
        return jax.numpy.square(vals).mean()

# ---------------------------------------------------------------

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_EM(typing.NamedTuple):
    
    sites_param: xt.iTuple
    sites_optimal: xt.iTuple # optimal at this step from em algo

    loc: xf.Location = None
    shape: xt.iTuple = None

    cut_tree: bool = False

    def apply(self, state):
        param = xf.concatenate_sites(self.sites_param, state)
        optimal = xf.concatenate_sites(self.sites_optimal, state)
        return loss_mse(
            param,
            ( 
                jax.lax.stop_gradient(optimal)
                if self.cut_tree
                else optimal
            )
        )

# ---------------------------------------------------------------

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_L0(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_L1(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_L2(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_ElasticNet(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_KernelVsCov(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        # param weights
        # y_pred = weights . normal
        # mse(y_pred, y)
        # 
        # cov_weights = weights.weights
        # cov_kernel = kernel(latent(x))
        # mse(cov_weights, cov_kernel)
        assert False, self


# ---------------------------------------------------------------

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_MaxSpread(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    T: bool = False

    def apply(self, state):
        data = xf.concatenate_sites(self.sites, state)
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

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_MSE(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert self.sites.len() == 2
        l_site, r_site = self.sites
        l = xf.get_location(l_site, state)
        r = xf.get_location(r_site, state)
        return loss_mse(l, r)

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_Orthonormal(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    T: bool = False

    def apply(self, state):
        X = xf.concatenate_sites(self.sites, state)
        if self.T:
            X = X.T
        return loss_orthonormal(X)

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_Orthogonal(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    T: bool = False

    def apply(self, state):
        X = xf.concatenate_sites(self.sites, state)
        if self.T:
            X = X.T
        return loss_orthogonal(X)

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_LinearCovar(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        site_l, site_r = self.sites
        X = xf.get_location(site_l, state)
        XXt = jax.numpy.matmul(X, X.T)
        cov = xf.get_location(site_r, state)
        return loss_mse(XXt, cov)

@xf.constraint_bindings()
@xt.nTuple.decorate
class Constraint_EigenVLike(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    eigval_max: bool = True

    n_check: int = None

    def apply(self, state):
        assert len(self.sites) == 2
        w_site, f_site = self.sites

        w = xf.get_location(w_site, state)
        f = xf.get_location(f_site, state)

        cov = jax.numpy.cov(f.T)
        eigvals = jax.numpy.diag(cov)

        if self.n_check is not None:
            assert eigvals.shape[0] == self.n_check, (
                self, eigvals.shape,
            )
        res = (
            + loss_descending(eigvals)
            + loss_orthogonal(w.T)
            + loss_mean_zero(0)(f)
            + loss_diag(cov, eigvals)
        )
        if self.eigval_max:
            return res + (
                - jax.numpy.sum(jax.numpy.log(1 + eigvals))
                # ridge penalty to counteract eigval max
            )
        return res
    
# ---------------------------------------------------------------
