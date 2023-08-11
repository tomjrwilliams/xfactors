
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
from ... import utils

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Weights_Constant(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    v: float
    shape: tuple

    

    def init_params(self, model, params):
        return self, jax.numpy.ones(self.shape) * self.v


@xt.nTuple.decorate()
class Weights_Normal(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    shape: tuple

    

    def init_params(self, model, params):
        return self, utils.rand.gaussian(self.shape)


@xt.nTuple.decorate()
class Weights_Orthogonal(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    shape: tuple

    

    def init_params(self, model, params):
        return self, utils.rand.orthogonal(self.shape)

# ---------------------------------------------------------------

def check_latent(obj, model):
    assert obj.axis in [None, 0, 1]
    return xf.check_operator(obj, model)

@xt.nTuple.decorate()
class Latent(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    n: int
    axis: int
    sites: xt.iTuple
    # TODO init: collections.abc.Iterable = None

    # kwargs for specifying the init - orthogonal, gaussian, etc.

    

    def init_params(self, model, params):
        axis = self.axis
        objs = self.sites.map(xf.f_get_location(model))
        shape_latent = (
            (self.n,)
            if axis is None
            else (
                objs.map(lambda o: o.shape[axis]).pipe(sum), 
                self.n,
            )
        )
        assert shape_latent is not None, self
        latent = utils.rand.gaussian(shape_latent)
        return self, latent

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)


# ---------------------------------------------------------------
