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

# NOTE: calc as classmethod so can also have a full gp operator that also does the sampling, without re-implementing the kernel 

# for the below, way to include vmap in the same class definition?
# or just have V_GP_... - probably simpler to do that.

@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_Constant(typing.NamedTuple):

    c: float

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self
    

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_Linear(typing.NamedTuple):

    sites: xt.iTuple

    # optional weights?
    # optional mean

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Linear(typing.NamedTuple):

    sites: xt.iTuple

    # optional weights?
    # optional mean

    n: int = None
    # shape determines how many samples
    # eg. if using a gp for three factors
    # if mean provided, assert mean.shape == n

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# TODO: VMap_... 

# given that we only have to do a single covar estimate per cls
# ie. we assume global, generate weights
# and then stack the weights up into pca esque matrix
# does actually make snese to have a latent gp?
# ah can still be done easily enough with split_by -> vmap
# that just only happens over the latent embeddings
# then re-group / flatten the vectors -> pca

# ---------------------------------------------------------------

# aka white noise
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_Gaussian(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self
       
# ---------------------------------------------------------------

# aka white noise
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_RBF(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------

# aka white noise
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_Sigmoid(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------
 
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_SquaredExp(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self
        
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_OU(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self
     
# ---------------------------------------------------------------
   
@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Kernel_RationalQuadratic(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------
