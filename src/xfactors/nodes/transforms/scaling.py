
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
from ... import utils
from .. import params


# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Expit(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Expit, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.expit(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Exp(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Exp, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.exp(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Sq(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Sq, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.square(self.data.access(state))

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Linear1D(typing.NamedTuple):

    a: xf.Loc
    b: xf.Loc
    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Linear1D, tuple, xf.SiteValue]: ...

    @classmethod
    def add_to_model(
        cls,
        model,
        data: xf.OptionalLoc = None,
        a: xf.OptionalLoc = None,
        b: xf.OptionalLoc = None,
    ):
        assert data is not None
        if a is None:
            model, a = model.add_node(params.random.Gaussian((1,)))
        if b is None:
            model, b = model.add_node(params.random.Gaussian((1,)))
        obj = cls(a=a, b=b, data=data)
        return model.add_node(obj)

    @classmethod
    def f(cls, data, a, b):
        return utils.funcs.linear(data, a, b)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(
            data=self.data.access(state),
            a=self.a.access(state),
            b=self.b.access(state),
        )

@xt.nTuple.decorate(init=xf.init_null)
class Logistic(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Logistic, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.logistic(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Sigmoid(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.sigmoid(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class CosineKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_cosine(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))


@xt.nTuple.decorate(init=xf.init_null)
class RBFKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_rbf(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class GaussianKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data = None
    ) -> tuple[Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_gaussian(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.Model,
        data = None,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

# ---------------------------------------------------------------
