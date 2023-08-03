
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

# from jax.config import config 
# config.update("jax_debug_nans", True) 

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class KMeans(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None


    def apply(self, state):
        # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
        # https://theory.stanford.edu/~sergei/papers/kMeans-socg.pdf
        data = jax.numpy.concatenate(
            self.sites.map(xf.f_get_location(state)),
            axis=1,
        )
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------
