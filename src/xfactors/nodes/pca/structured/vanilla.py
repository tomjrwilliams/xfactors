


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

from .... import utils
from .... import xfactors as xf

# ---------------------------------------------------------------

# weight mask to apply to loadings directly
# eg. where we have a flat dataframe of two curves
# btu want factors that are only one or the other


@xt.nTuple.decorate()
class Structured_PCA_Mask(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    

    def apply(self, state):
        return

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Structured_PCA_Convex(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    

    def apply(self, state):
        return



@xt.nTuple.decorate()
class Structured_PCA_Concave(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    

    def apply(self, state):
        return


# ---------------------------------------------------------------

# overall factor sign
# has the same effect as the below (but if we don't care oither than the same can use below)

@xt.nTuple.decorate()
class Structured_PCA_Sign(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    

    def apply(self, state):
        return


# eg. just for factor alignment

@xt.nTuple.decorate()
class Structured_PCA_TiedSign(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    

    def apply(self, state):
        return

# ---------------------------------------------------------------
