
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

def loss_cov_diag(cov, diag):
    diag = jax.numpy.multiply(
        jax.numpy.eye(cov.shape[0]), diag
    )
    return loss_mse(cov, diag)

def loss_orthogonal(X, scale = 1.):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T) / scale
    # return jax.numpy.square(jax.numpy.subtract(XXt, eye)).mean()
    return loss_mse(XXt, eye)

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
class Constraint_Orthogonal(typing.NamedTuple):
    
    site: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        X = xf.get_location(self.site, state)
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
            + loss_cov_diag(cov, eigvals)
        )
        if self.eigval_max:
            return res + (
                - jax.numpy.sum(jax.numpy.log(1 + eigvals))
                # ridge penalty to counteract eigval max
            )
        return res
    
# ---------------------------------------------------------------
