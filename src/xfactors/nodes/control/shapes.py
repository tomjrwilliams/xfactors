
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



@xt.nTuple.decorate()
class Stack(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...    

    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self

# flatten such a tuple from the above back down



@xt.nTuple.decorate()
class UnStack(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...

    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------



@xt.nTuple.decorate()
class Flatten(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self


@xt.nTuple.decorate()
class UnFlatten(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self

# flatten such a tuple from the above back down
# versus the above which actually concats the arrays

# ---------------------------------------------------------------



@xt.nTuple.decorate()
class Concatenate(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self


# given shape definitions, can slice back out into tuple

@xt.nTuple.decorate()
class UnConcatenate(typing.NamedTuple):
    
    sites_values: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
