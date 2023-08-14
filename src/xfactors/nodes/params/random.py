
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
from ... import utils

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class RandomCovariance(typing.NamedTuple):

    n: int
    d: int

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[RandomCovariance, tuple, tuple]:
        shape = (self.d, self.d)
        gaussians = [
            utils.rand.gaussian(shape=shape)
            for i in range(self.n)
        ]
        return self, shape, jax.numpy.stack([
            jax.numpy.matmul(g.T, g)
            for g in gaussians
        ])

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return xf.get_location(site.loc.as_param(), state)

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Gaussian(typing.NamedTuple):

    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Gaussian, tuple, tuple]:
        return self,self.shape,  utils.rand.gaussian(self.shape)

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return xf.get_location(site.loc.as_param(), state)


@xt.nTuple.decorate()
class GaussianSoftmax(typing.NamedTuple):

    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[GaussianSoftmax, tuple, tuple]:
        return self, self.shape, jax.nn.softmax(
            utils.rand.gaussian(self.shape),
            axis=-1
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return xf.get_location(site.loc.as_param(), state)

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Beta(typing.NamedTuple):

    a: float
    b: float
    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Beta, tuple, tuple]:
        return self, self.shape, utils.rand.beta(self.a, self.b, self.shape)

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return xf.get_location(site.loc.as_param(), state)

# ---------------------------------------------------------------
