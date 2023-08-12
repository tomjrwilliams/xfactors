
from __future__ import annotations
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
from ... import xfactors as xf

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


@xt.nTuple.decorate()
class Kernel_Sum(typing.NamedTuple):

    c: float

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Sum, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    


@xt.nTuple.decorate()
class Kernel_Product(typing.NamedTuple):

    # take others as fields
    # but hence assume that sites are compatible

    c: float

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Product, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Kernel_Constant(typing.NamedTuple):

    c: float

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Constant, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Kernel_Linear(typing.NamedTuple):

    sites: xt.iTuple

    # optional weights?
    # optional mean

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Linear, tuple, tuple]: ...
    
    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        # norms = jax.numpy.sum(
        #     jax.numpy.multiply(features_l, features_r),
        #     axis=1
        # )
        norms = euclidean_distance(features_l, features_r)
        return norms

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Kernel_Gaussian(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Gaussian, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
       
# ---------------------------------------------------------------



@xt.nTuple.decorate()
class Kernel_RBF(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    # TODO: optional transform callable field
    # that can take params itself

    # so eg. can transform with |x - center| 
    # for eg. rate tenor kernel pca

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_RBF, tuple, tuple]: ...
    
    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        l_2_sq = 2 * jax.numpy.square(l)
        norms = euclidean_distance(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l_2_sq)
        ) * sigma_sq

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Kernel_Sigmoid(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Sigmoid, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
 

@xt.nTuple.decorate()
class Kernel_SquaredExp(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_SquaredExp, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
        

@xt.nTuple.decorate()
class Kernel_OU(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_OU, tuple, tuple]: ...
    
    @classmethod
    def f(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        # l_2_sq = 2 * jax.numpy.square(l)
        norms = euclidean_distance(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l)
        ) * sigma_sq

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
     
# ---------------------------------------------------------------
   

@xt.nTuple.decorate()
class Kernel_RationalQuadratic(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_RationalQuadratic, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
