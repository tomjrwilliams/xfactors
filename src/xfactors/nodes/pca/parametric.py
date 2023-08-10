
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

from . import pca

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class Parametric_Factor(typing.NamedTuple):
    
    features: xt.iTuple
    params: xt.iTuple

    # or an operator for the function, which can have its own params site? probably that

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):

        # given a feature matrix 
        # eg. simple one is a constant column per tenor (of the tenor represented as a float)
        # for rates parametric factor pca
        
        # how to control if we have to apply to a whole matrix or just a vector?
        # eg the tenor can be a single global vector

        # then we just have to scale it each day by the latent factor for the intensity of that factor


        # can have separate vector wise parametric functions
        # that we have a stack operator to join into singl eloadings matrix

        # and then that can be multiplied to the latent factor path
        # latent as opposed to encoded
        # -> yields


        # apply the functions callables loading into the tuple

        return