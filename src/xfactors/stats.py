
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

from . import rand
from . import dates
from . import xfactors as xf

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class Cov(typing.NamedTuple):

    sites: xt.iTuple

    # ---

    random: bool = False
    static: bool = False
    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        objs = self.sites.map(xf.f_get_location(model))
        n = objs.map(lambda o: o.shape[1]).pipe(sum)
        return self._replace(
            shape = (n, n,),
        )

    def apply(self, state):
        data = jax.numpy.concatenate(
            self.sites.map(xf.f_get_location(state)),
            axis=1,
        )
        return jax.numpy.cov(
            jax.numpy.transpose(data)
        )

# ---------------------------------------------------------------
