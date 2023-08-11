
from __future__ import annotations
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

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Input_DataFrame_Wide, tuple, tuple]:
        # path[0] = stage, so path[1] = index of data element
        df = data[self.loc.path[1]]
        return self._replace(
            **({} if not self.fixed_columns else dict(
                columns=xt.iTuple(df.columns)
            )),
            **({} if not self.fixed_index else dict(
                index=xt.iTuple(df.index)
            )),
        ), df.values.shape, ()
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
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

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Input_DataFrame_Tall, tuple, tuple]:
        # path[0] = stage, so path[1] = index of data element
        return self, data[self.loc.path[1]].values.shape, ()
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        _, data, _, _ = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------
