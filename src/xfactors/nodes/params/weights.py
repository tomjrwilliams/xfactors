
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
class Gaussian(typing.NamedTuple):

    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Gaussian, tuple, xf.SiteValue]:
        shape = (
            self.data.access(model, into=xf.Site).shape[1],
            self.n,
        )
        return self, shape, utils.rand.gaussian(shape)

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

    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VGaussian, tuple, xf.SiteValue]:
        shape = (
            self.data.access(model).shape.map(
                lambda s: (s[1], self.n,)
            )
        )
        return self, shape, shape.map(utils.rand.gaussian)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        v = site.loc.as_param().access(state, into=xt.iTuple)
        if site.masked:
            return v.map(jax.lax.stop_gradient)
        return v

@xt.nTuple.decorate()
class VOrthogonal(typing.NamedTuple):

    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VOrthogonal, tuple, xf.SiteValue]:
        shape = xt.iTuple(
            self.data.access(model, into=xf.Site).shape
        ).map(
            lambda s: (s[1], self.n,)
        )
        return self, shape, shape.map(
            lambda s: utils.rand.orthogonal(s[0])[..., :s[1]]
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
class ConcatenateGaussian(typing.NamedTuple):

    n: int
    data: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[ConcatenateGaussian, tuple, xf.SiteValue]:
        data = self.data.map(
            lambda s: s.access(model, into=xf.Site).shape
        )
        shape = data.flatten()
        return self, shape, utils.rand.gaussian(shape)

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
