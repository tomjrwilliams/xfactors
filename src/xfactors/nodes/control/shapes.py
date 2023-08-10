
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

# below is eg. for latent gp model
# so we first group by (say) sector
# then for each, vmap gp
# then flatten back down & apply constraints (mse, ...)


# ---------------------------------------------------------------


@xf.operator_bindings()
@xt.nTuple.decorate
class Stack(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# flatten such a tuple from the above back down


@xf.operator_bindings()
@xt.nTuple.decorate
class UnStack(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------


@xf.operator_bindings()
@xt.nTuple.decorate
class Flatten(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.operator_bindings()
@xt.nTuple.decorate
class UnFlatten(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# flatten such a tuple from the above back down
# versus the above which actually concats the arrays

# ---------------------------------------------------------------


@xf.operator_bindings()
@xt.nTuple.decorate
class Concatenate(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self


# given shape definitions, can slice back out into tuple
@xf.operator_bindings()
@xt.nTuple.decorate
class UnConcatenate(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
