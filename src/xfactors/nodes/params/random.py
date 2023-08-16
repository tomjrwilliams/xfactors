
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
    ) -> tuple[RandomCovariance, tuple, xf.SiteValue]:
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
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        
        v = site.loc.as_param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Gaussian(typing.NamedTuple):

    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Gaussian, tuple, xf.SiteValue]:
        return self,self.shape,  utils.rand.gaussian(self.shape)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        
        v = site.loc.as_param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v

@xt.nTuple.decorate()
class VGaussian(typing.NamedTuple):

    data: xf.Location
    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VGaussian, tuple, xf.SiteValue]:
        shape = xt.iTuple(self.data.access(model, into=xf.Site))
        return self, (
            shape.map(lambda _: self.shape)
        ), (
            shape.map(lambda _: utils.rand.gaussian(self.shape))
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        
        v = site.loc.as_param().access(state)
        if site.masked:
            return v.map(jax.lax.stop_gradient)
        return v

@xt.nTuple.decorate()
class GaussianSoftmax(typing.NamedTuple):

    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[GaussianSoftmax, tuple, xf.SiteValue]:
        return self, self.shape, jax.nn.softmax(
            utils.rand.gaussian(self.shape),
            axis=-1
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        
        v = site.loc.as_param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Beta(typing.NamedTuple):

    a: float
    b: float
    shape: tuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Beta, tuple, xf.SiteValue]:
        return self, self.shape, utils.rand.beta(self.a, self.b, self.shape)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        
        v = site.loc.as_param().access(state)
        if site.masked:
            return jax.lax.stop_gradient(v)
        return v

# ---------------------------------------------------------------
