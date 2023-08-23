
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
from ... import utils

# ---------------------------------------------------------------

# TODO: if eg. learning factor path over specific dates
# then here is where we encode that restriction
# specific stock universe, etc.


@xt.nTuple.decorate()
class Input_NDArray(typing.NamedTuple):

    nd: numpy.ndarray

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Input_NDArray, tuple, xf.SiteValue]:
        return self, self.nd.shape, ()
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return jax.numpy.array(self.nd)

# ---------------------------------------------------------------
