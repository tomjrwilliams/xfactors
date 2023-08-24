
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

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_Maximise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_Maximise, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return -1 * data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_Minimise(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_Minimise, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return data.mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_MinimiseSquare(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_MinimiseSquare, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L0(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_L0, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L1(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_L1, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.abs(data).mean()

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_VL1(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_L1, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.abs(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_L2(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_L2, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(data).mean()

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_VL2(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_L2, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = xt.ituple(self.data.access(state))
        return jax.numpy.vstack(data.map(
            lambda v: jax.numpy.square(v).mean()
        ).pipe(list)).mean()


@xt.nTuple.decorate(init=xf.init_null)
class Constraint_ElasticNet(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_ElasticNet, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_MAbsE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return utils.funcs.loss_mabse(l, r)

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_VMAbsE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mabse, r).pipe(list)
        ).mean()
# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_MSE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return utils.funcs.loss_mse(l, r)

@xt.nTuple.decorate(init=xf.init_null)
class Constraint_VMSE(typing.NamedTuple):
    
    l: xf.Location
    r: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Constraint_MSE, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        l = self.l.access(state)
        r = self.r.access(state)
        return jax.numpy.vstack(
            xt.ituple(l).map(utils.funcs.loss_mse, r).pipe(list)
        ).mean()

# ---------------------------------------------------------------
