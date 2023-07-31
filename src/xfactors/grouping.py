
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
class Group_By(typing.NamedTuple):
    
    sites_values: xt.iTuple
    sites_keys: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def apply(self, state):
        assert False, self

@xf.operator_bindings()
@xt.nTuple.decorate
class Flatten(typing.NamedTuple):
    
    sites_values: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# flatten such a tuple from the above back down

# ---------------------------------------------------------------
