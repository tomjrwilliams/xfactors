
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
class Scalar(typing.NamedTuple):

    v: numpy.ndarray

    

    def init_params(self, model, params):
        return self, jax.numpy.array(self.v)

    def apply(self, site: xf.Site, state: tuple) -> tuple:
        return xf.get_location(self.loc.as_param(), state)


# ---------------------------------------------------------------