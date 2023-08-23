
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
class Scale_Expit(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Expit, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.scipy.special.expit(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Exp(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Exp, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.exp(self.data.access(state))
@xt.nTuple.decorate(init=xf.init_null)
class Scale_Sq(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sq, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.square(self.data.access(state))
# ---------------------------------------------------------------
