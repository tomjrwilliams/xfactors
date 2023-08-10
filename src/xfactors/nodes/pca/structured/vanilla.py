


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

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class Structured_PCA_Convex(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        return


@xf.operator_bindings()
@xt.nTuple.decorate
class Structured_PCA_Concave(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        return


# ---------------------------------------------------------------

# overall factor sign
# has the same effect as the below (but if we don't care oither than the same can use below)
@xf.operator_bindings()
@xt.nTuple.decorate
class Structured_PCA_Sign(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        return


# eg. just for factor alignment
@xf.operator_bindings()
@xt.nTuple.decorate
class Structured_PCA_TiedSign(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        return

# ---------------------------------------------------------------
