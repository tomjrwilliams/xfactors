
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

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Expit(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Expit, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return jax.scipy.special.expit(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Exp(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Exp, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.exp(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Sq(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sq, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return jax.numpy.square(self.data.access(state))

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Linear1D(typing.NamedTuple):

    a: xf.Loc
    b: xf.Loc
    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Linear1D, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data, a, b):
        return (data * b) + a

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(
            data=self.data.access(state),
            a=self.a.access(state),
            b=self.b.access(state),
        )

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Logistic(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Logistic, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return 1 / (
            jax.numpy.exp(data) + 2 + jax.numpy.exp(-data)
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Sigmoid(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return (2 / numpy.pi) * (
            1 / (jax.numpy.exp(data) + jax.numpy.exp(-data))
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_CosineKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        data = Scale_Expit.f(data)
        return (numpy.pi / 4) * (
            jax.numpy.cos(
                (numpy.pi / 2) * data
            )
        )

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Expit_Linear1D(typing.NamedTuple):

    data: xf.Loc
    a: xf.Loc
    b: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Expit_Linear1D, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        a = self.a.access(state)
        b = self.b.access(state)
        return Scale_Linear1D.f(
            data=Scale_Expit.f(data), a=a, b=b,
        )

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Sigmoid_Linear1D(typing.NamedTuple):

    data: xf.Loc
    a: xf.Loc
    b: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sigmoid_Linear1D, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        a = self.a.access(state)
        b = self.b.access(state)
        return Scale_Linear1D.f(
            data=Scale_Sigmoid.f(data), a=a, b=b,
        )

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Logistic_Linear1D(typing.NamedTuple):

    data: xf.Loc
    a: xf.Loc
    b: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Logistic_Linear1D, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        a = self.a.access(state)
        b = self.b.access(state)
        return Scale_Linear1D.f(
            data=Scale_Logistic.f(data), a=a, b=b,
        )

@xt.nTuple.decorate(init=xf.init_null)
class Scale_CosineKernel_Linear1D(typing.NamedTuple):

    data: xf.Loc
    a: xf.Loc
    b: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_CosineKernel_Linear1D, tuple, xf.SiteValue]: ...

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        a = self.a.access(state)
        b = self.b.access(state)
        return Scale_Linear1D.f(
            data=Scale_CosineKernel.f(data), a=a, b=b,
        )

# ---------------------------------------------------------------
