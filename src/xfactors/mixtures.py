
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
class GMM(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

@xf.operator_bindings()
@xt.nTuple.decorate
class BGMM(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None


    def apply(self, state):
        # https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf
        # https://scikit-learn.org/0.15/modules/dp-derivation.html

        data = xf.concatenate_sites(self.sites, state, axis = 1)

        return

# ---------------------------------------------------------------
