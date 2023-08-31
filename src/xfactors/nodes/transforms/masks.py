
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

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Zero(typing.NamedTuple):

    data: xf.Loc
    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Zero, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return jax.numpy.multiply(
            self.data.access(state),
            self.v
        )


@xt.nTuple.decorate(init=xf.init_null)
class Positive(typing.NamedTuple):

    data: xf.Loc
    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Positive, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        data_pos = jax.numpy.abs(data)
        pos_mask = self.v
        no_mask = 1 + (self.v * -1)
        return jax.numpy.multiply(
            data, no_mask
        ) + jax.numpy.multiply(
            data_pos, pos_mask
        )
@xt.nTuple.decorate(init=xf.init_null)
class Negative(typing.NamedTuple):

    data: xf.Loc
    v: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Negative, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        data_pos = -1 * jax.numpy.abs(data)
        pos_mask = self.v
        no_mask = 1 + (self.v * -1)
        return jax.numpy.multiply(
            data, no_mask
        ) + jax.numpy.multiply(
            data_pos, pos_mask
        )
# ---------------------------------------------------------------
