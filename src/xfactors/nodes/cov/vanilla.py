
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
class Cov(typing.NamedTuple):

    data: xf.Location

    # ---

    random: bool = False
    static: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Cov, tuple, xf.SiteValue]:
        vs = self.data.site().access(model)
        n = vs.shape[1]
        return self, (n, n,), ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        res = jax.numpy.cov(
            jax.numpy.transpose(data)
        )
        return res

@xt.nTuple.decorate(init=xf.init_null)
class VCov(typing.NamedTuple):

    data: xf.Location

    # ---

    random: bool = False
    static: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VCov, tuple, xf.SiteValue]:
        vs = self.data.site().access(model)
        shape = vs.map(lambda v: v.shape[1]).map(lambda n: (n, n,))
        return self, shape, ()

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        res = data.map(lambda vs: jax.numpy.cov(
            jax.numpy.transpose(vs)
        ))
        return res
        
# TODO move shrinkage here

# also add cov_with_missing where we have different number of samples not none in a given df
# so pairwise calculate with different denominator (with a minimin sampling number)


# can even then have shrinkage maybe calcualted per name?

# ---------------------------------------------------------------
