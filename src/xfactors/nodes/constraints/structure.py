
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
class Constraint_Positive(typing.NamedTuple):
    
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_Positive, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        return jax.numpy.square(jax.numpy.clip(data, a_max = 0)).mean()

# ---------------------------------------------------------------
