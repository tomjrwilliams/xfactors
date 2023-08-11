
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

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Scalar(typing.NamedTuple):

    v: numpy.ndarray

    

    def init_params(self, model, params):
        return self, jax.numpy.array(self.v)

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)


# ---------------------------------------------------------------


@xt.nTuple.decorate()
class RandomCovariance(typing.NamedTuple):

    n: int
    d: int

    

    def init_params(self, model, params):
        gaussians = [
            rand.gaussian(shape=(self.d, self.d))
            for i in range(self.n)
        ]
        return self, jax.numpy.stack([
            jax.numpy.matmul(g.T, g)
            for g in gaussians
        ])

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Gaussian(typing.NamedTuple):

    shape: tuple

    

    def init_params(self, model, params):
        return self, rand.gaussian(self.shape)

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)


@xt.nTuple.decorate()
class GaussianSoftmax(typing.NamedTuple):

    shape: tuple

    

    def init_params(self, model, params):
        return self, jax.nn.softmax(
            rand.gaussian(self.shape),
            axis=-1
        )

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Beta(typing.NamedTuple):

    shape: tuple

    

    def init_params(self, model, params):
        return self, rand.beta(self.shape)

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)

# ---------------------------------------------------------------
