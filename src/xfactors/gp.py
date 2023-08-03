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

def euclidean_distance(l, r, small = 10 ** -3):
    diffs_sq = jax.numpy.square(jax.numpy.subtract(l, r))
    return jax.numpy.sqrt(
        jax.numpy.sum(diffs_sq, axis = -1) + small
    )

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

    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        # norms = jax.numpy.sum(
        #     jax.numpy.multiply(features_l, features_r),
        #     axis=1
        # )
        norms = euclidean_distance(features_l, features_r)
        return norms

    def apply(self, state):
        assert False, self

@xf.operator_bindings()
@xt.nTuple.decorate
class GP_Linear(typing.NamedTuple):

    sites_features: xt.iTuple
    sites_data: xt.iTuple

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

    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        l_2_sq = 2 * jax.numpy.square(l)
        norms = euclidean_distance(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l_2_sq)
        ) * sigma_sq

    def apply(self, state):
        assert False, self


@xf.operator_bindings()
@xt.nTuple.decorate
class GP_RBF(typing.NamedTuple):

    # sigma: float
    sites_features: xt.iTuple
    sites_data: xt.iTuple

    # optional weights?
    # optional mean

    n: int = None
    # shape determines how many samples
    # eg. if using a gp for three factors
    # if mean provided, assert mean.shape == n

    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_params(self, model, params):
        return self, (
            jax.numpy.ones(1),
            jax.numpy.ones(1),
        )

    def apply(self, state):

        l, sigma = xf.get_location(self.loc.as_param(), state)
        
        # = n_variables, n_latents
        features = xf.concatenate_sites(
            self.sites_features, state, axis = 1
        )

        n_variables = features.shape[0]
        n_features = features.shape[1]

        data = xf.concatenate_sites(
            self.sites_data, state, axis = 1
        )

        assert data.shape[1] == n_variables

        features_matrix = jax.numpy.resize(jax.numpy.expand_dims(
            features, axis=0
        ), (features.shape[0], *features.shape,))

        features_l = jax.numpy.reshape(
            features_matrix,
            (n_variables ** 2, n_features,)
        )
        # left = iterates through variables
        features_r = jax.numpy.reshape(
            jax.numpy.transpose(features_matrix, (1, 0, 2,)),
            (n_variables ** 2, n_features,)
        )
        # right = blocks of same variable's latents
        kernel = GP_Kernel_RBF.f(
            features_r,
            features_l,
            l,
            sigma,
        )
        cov = jax.numpy.reshape(
            kernel, (n_variables, n_variables,)
        )

        # assert (cov == cov.T).all()

        return cov

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

    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        # l_2_sq = 2 * jax.numpy.square(l)
        norms = euclidean_distance(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l)
        ) * sigma_sq

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
