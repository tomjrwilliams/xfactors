
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


@xt.nTuple.decorate()
class Cov(typing.NamedTuple):

    sites: xt.iTuple

    # ---

    random: bool = False
    static: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA, tuple, tuple]: ...
    
    def init_shape(self, site, model, data):
        objs = self.sites.map(xf.f_get_location(model))
        n = objs.map(lambda o: o.shape[1]).pipe(sum)
        return self._replace(
            shape = (n, n,),
        )

    def apply(self, site: xf.Site, state: tuple) -> tuple:
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        res = jax.numpy.cov(
            jax.numpy.transpose(data)
        )
        return res

# TODO move shrinkage here

# also add cov_with_missing where we have different number of samples not none in a given df
# so pairwise calculate with different denominator (with a minimin sampling number)


# can even then have shrinkage maybe calcualted per name?

# ---------------------------------------------------------------
