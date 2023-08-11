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
from ... import xfactors as xf

# ---------------------------------------------------------------

# TODO: if eg. learning factor path over specific dates
# then here is where we encode that restriction
# specific stock universe, etc.


@xt.nTuple.decorate()
class Input_DataFrame_Wide(typing.NamedTuple):

    fixed_columns: bool = False
    fixed_index: bool = False

    columns: xt.iTuple = None
    index: xt.iTuple = None

    def init_shape(self, site, model, data):
        # path[0] = stage, so path[1] = index of data element
        df = data[self.loc.path[1]]
        return self._replace(
            shape=df.values.shape,
            **({} if not self.fixed_columns else dict(
                columns=xt.iTuple(df.columns)
            )),
            **({} if not self.fixed_index else dict(
                index=xt.iTuple(df.index)
            )),
        )
    
    def apply(self, state):
        _, data, _, _ = state
        df = data[self.loc.path[-1]]
        if self.fixed_columns:
            columns = xt.iTuple(df.columns)
            assert columns == self.columns, dict(
                fixed=self.columns,
                given=columns
            )
        if self.fixed_index:
            index = xt.iTuple(df.index)
            assert index == self.index, dict(
                fixed=self.index,
                given=index
            )
        return jax.numpy.array(df.values)


@xt.nTuple.decorate()
class Input_DataFrame_Tall(typing.NamedTuple):

    # fields to specify if keep index and ticker map

    def init_shape(self, site, model, data):
        # path[0] = stage, so path[1] = index of data element
        return self._replace(
            shape=data[self.loc.path[1]].values.shape
        )
    
    def apply(self, state):
        _, data, _, _ = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------
