
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

    columns_: typing.Optional[xt.iTuple] = None
    index_: typing.Optional[xt.iTuple] = None

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Input_DataFrame_Wide, tuple, tuple]:
        assert site.loc is not None
        # path[0] = stage, so path[1] = index of data element
        df = data[site.loc.path[1]]
        if self.fixed_columns:
            self = self._replace(columns_ = xt.iTuple(df.columns))
        if self.fixed_index:
            self = self._replace(index_ = xt.iTuple(df.index))
        return self, df.values.shape, ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        data = state.data
        df = data[site.loc.path[-1]]
        if self.fixed_columns:
            columns = xt.iTuple(df.columns)
            assert columns == self.columns_, dict(
                fixed=self.columns_,
                given=columns
            )
        if self.fixed_index:
            index = xt.iTuple(df.index)
            assert index == self.index_, dict(
                fixed=self.index_,
                given=index
            )
        return jax.numpy.array(df.values)


@xt.nTuple.decorate()
class Input_DataFrame_Tall(typing.NamedTuple):

    # fields to specify if keep index and ticker map

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Input_DataFrame_Tall, tuple, tuple]:
        assert site.loc is not None
        # path[0] = stage, so path[1] = index of data element
        return self, data[site.loc.path[1]].values.shape, ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        data = state.data
        df = data[site.loc.path[-1]]
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------
