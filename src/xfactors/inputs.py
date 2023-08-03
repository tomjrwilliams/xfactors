import enum

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

# TODO: if eg. learning factor path over specific dates
# then here is where we encode that restriction
# specific stock universe, etc.

@xf.input_bindings()
@xt.nTuple.decorate
class Input_DataFrame_Wide(typing.NamedTuple):

    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        # path[0] = stage, so path[1] = index of data element
        return self._replace(
            shape=data[self.loc.path[1]].values.shape
        )
    
    def apply(self, state):
        _, data, _, _ = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

@xf.input_bindings()
@xt.nTuple.decorate
class Input_DataFrame_Tall(typing.NamedTuple):

    # fields to specify if keep index and ticker map

    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        # path[0] = stage, so path[1] = index of data element
        return self._replace(
            shape=data[self.loc.path[1]].values.shape
        )
    
    def apply(self, state):
        _, data, _, _ = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------
