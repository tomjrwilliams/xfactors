
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
class Scalar(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    v: numpy.ndarray

    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_params(self, model, params):
        return self, jax.numpy.array(self.v)

    def apply(self, state):
        return xf.get_location(self.loc.as_param(), state)


# ---------------------------------------------------------------
