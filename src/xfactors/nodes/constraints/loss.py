
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

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_Maximise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_Maximise, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return -1 * data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_Minimise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_Minimise, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_MinimiseSquare(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MinimiseSquare, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L0(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_L0, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L1(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_L1, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(data).mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L2(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_L2, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_ElasticNet(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_ElasticNet, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_MSE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_MSE, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return funcs.loss_mse(l, r)


# ---------------------------------------------------------------
