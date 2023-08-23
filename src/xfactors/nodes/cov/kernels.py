
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


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Sum(typing.NamedTuple):

    kernels: xt.iTuple = xt.iTuple()
    kernel: typing.Optional[xf.Loc] = None

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Sum, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        if len(self.kernels):
            kernels = self.kernels.map(
                lambda k: k.access(state)
            ).map(
                lambda v: (
                    v if isinstance(v, xt.iTuple) else xt.iTuple((v,))
                )
            ).flatten()
        else:
            kernels = xt.iTuple()
        if self.kernel is not None:
            kernels = kernels.extend(self.kernel.access(state))
        agg = jax.numpy.stack(list(kernels))
        return agg.sum(axis=0)
    


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Product(typing.NamedTuple):

    # take others as fields
    # but hence assume that sites are compatible

    c: float

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Product, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Constant(typing.NamedTuple):

    c: float

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Constant, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
    

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Linear(typing.NamedTuple):

    a: xf.Loc
    c: xf.Loc
    sigma: xf.Loc
    data: xf.Loc

    sigma_sq: bool = True

    # assumed 1D

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Linear, tuple, xf.SiteValue]: ...
    
    @classmethod
    def f(cls, data, a, c, sigma, sigma_sq = True):

        diffs = data - c
        diffs_ = xf.expand_dims(diffs, 0, 1)
        # 1, n_points

        # scale

        return (
            (diffs_ * diffs_.T) * (
                jax.numpy.square(sigma)
                if sigma_sq
                else sigma
            )
        ) + jax.numpy.square(a)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, a, c, sigma, sigma_sq=self.sigma_sq)


@xt.nTuple.decorate(init=xf.init_null)
class VKernel_Linear(typing.NamedTuple):

    a: xf.Loc
    c: xf.Loc
    sigma: xf.Loc
    data: xf.Loc

    sigma_sq: bool = True

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VKernel_Linear, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return xt.iTuple(a).zip(c, sigma).mapstar(
            lambda _a, _c, _sigma: Kernel_Linear.f(data, _a, _c, _sigma, sigma_sq= self.sigma_sq)
        )


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_VLinear(typing.NamedTuple):

    a: xf.Loc
    c: xf.Loc
    sigma: xf.Loc
    data: xf.Loc

    sigma_sq: bool = True

    # assumed 1D

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_VLinear, tuple, xf.SiteValue]: ...
    
    @classmethod
    def f(cls, data, a, c, sigma, sigma_sq = True):

        diffs = (xf.expand_dims(data, 0, 1) - (
            xf.expand_dims(c, 0, 1).T
        )).T
        # n_points, n_c

        diffs_ = xf.expand_dims(diffs, 0, diffs.shape[0])

        diff_prod = (
            diffs_ * jax.numpy.transpose(
                diffs_, (1, 0, 2),
            )
        )

        diff_soft = jax.nn.softmax(diff_prod, axis = 2)
        diff_weights = 1 - diff_soft

        return (
            # diff_prod.mean(axis=-1) * (
            (diff_weights * diff_prod).sum(axis=-1) * (
                jax.numpy.square(sigma)
                if sigma_sq
                else sigma
            )
        ) + jax.numpy.square(a)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return self.f(data, a, c, sigma, sigma_sq=self.sigma_sq)


@xt.nTuple.decorate(init=xf.init_null)
class VKernel_VLinear(typing.NamedTuple):

    a: xf.Loc
    c: xf.Loc
    sigma: xf.Loc
    data: xf.Loc

    sigma_sq: bool = True

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[VKernel_VLinear, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        a = self.a.access(state)
        c = self.c.access(state)
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        return xt.iTuple(a).zip(c, sigma).mapstar(
            lambda _a, _c, _sigma: Kernel_VLinear.f(data, _a, _c, _sigma, sigma_sq = self.sigma_sq)
        )
# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Gaussian(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Gaussian, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
       
# ---------------------------------------------------------------

small = 10 ** -4


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_RBF(typing.NamedTuple):

    sigma: xf.Loc
    l: xf.Loc
    data: xf.Loc

    # TODO: optional transform callable field
    # that can take params itself

    # so eg. can transform with |x - center| 
    # for eg. rate tenor kernel pca

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_RBF, tuple, xf.SiteValue]: ...
    
    @classmethod
    def f_flat(cls, features_l, features_r, sigma, l):
        sigma_sq = jax.numpy.square(sigma)
        l_2_sq = 2 * jax.numpy.square(l)
        norms = euclidean_distance(features_l, features_r)
        return jax.numpy.exp(
            -1 * (jax.numpy.square(norms) / l_2_sq)
        ) * sigma_sq

    @classmethod
    def f(cls, data, sigma, l):

        sigma_sq = jax.numpy.square(sigma)
        l_2_sq = 2 * jax.numpy.square(l)

        data_ = xf.expand_dims(data, 0, 1)
        diffs = data_ - data_.T
        diffs_sq = jax.numpy.square(diffs)

        # prevent div 0
        euclidean = jax.numpy.sqrt(diffs_sq + small)

        return jax.numpy.exp(
            -1 * (jax.numpy.square(euclidean) / l_2_sq)
        ) * sigma_sq

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        sigma = self.sigma.access(state)
        data = self.data.access(state)
        l = self.l.access(state)
        return self.f(data, sigma, l)


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class Kernel_Sigmoid(typing.NamedTuple):

    sigma: float
    # or variance?
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_Sigmoid, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
 

@xt.nTuple.decorate(init=xf.init_null)
class Kernel_SquaredExp(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_SquaredExp, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
        

@xt.nTuple.decorate(init=xf.init_null)
class Kernel_OU(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_OU, tuple, xf.SiteValue]: ...
    
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
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self
     
# ---------------------------------------------------------------
   

@xt.nTuple.decorate(init=xf.init_null)
class Kernel_RationalQuadratic(typing.NamedTuple):

    length_scale: float
    sites: xt.iTuple

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Kernel_RationalQuadratic, tuple, xf.SiteValue]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
