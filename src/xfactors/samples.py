
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
class Latent_Gaussian(typing.NamedTuple):
    
    cls_sites: xt.iTuple
    mu_sites: xt.iTuple
    cov_sites: xt.iTuple
    # should be all the same length

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False
        
# ---------------------------------------------------------------
