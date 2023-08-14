
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

# below is eg. for latent gp model
# so we first group by (say) sector
# then for each, vmap gp
# then flatten back down & apply constraints (mse, ...)


# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Stack(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Stack, tuple, tuple]: ...    

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down



@xt.nTuple.decorate(init=xf.init_null)
class UnStack(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[UnStack, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Flatten(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Flatten, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class UnFlatten(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[UnFlatten, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down
# versus the above which actually concats the arrays

# ---------------------------------------------------------------



@xt.nTuple.decorate(init=xf.init_null)
class Concatenate(typing.NamedTuple):
    
    locs: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Concatenate, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


# given shape definitions, can slice back out into tuple

@xt.nTuple.decorate(init=xf.init_null)
class UnConcatenate(typing.NamedTuple):
    
    loc: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[UnConcatenate, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
