
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

def fold_f_into(obj, fs: xt.iTuple):
    assert isinstance(obj, typing.NamedTuple), obj
    f = obj.f
    def f_new(self, *args, **kwargs):
        res = f(*args, **kwargs)
        return fs.fold(
            lambda acc, _f: _f(acc), initial=res
        )
    setattr(obj, "f", f_new)
    return obj

# ---------------------------------------------------------------

@xt.nTuple.decorate(init=xf.init_null)
class Scale_Expit(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Expit, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.expit(data)

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
    def add_to_model(
        cls,
        model,
        stage,
        PARAMS = None,
        data: xf.OptionalLoc = None,
        a: xf.OptionalLoc = None,
        b: xf.OptionalLoc = None,
        pipe_into: typing.Optional[xt.iTuple]=None,
    ):
        assert "data" is not None
        if a is None:
            model = model.add_node(
                PARAMS, params.random.Gaussian((1,))
            )
            a = model.last_added.as_param()
        if b is None:
            model = model.add_node(
                PARAMS, params.random.Gaussian((1,))
            )
            b = model.last_added.as_param()
        obj = cls(a=a, b=b, data=data)
        if pipe_into is not None:
            obj = fold_f_into(obj, pipe_into)
        return model.add_node(stage, obj)

    @classmethod
    def f(cls, data, a, b):
        return utils.funcs.expit(data, a, b)

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
        return utils.funcs.logistic(data)

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
        return utils.funcs.sigmoid(data)

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
        return utils.funcs.kernel_cosine(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))


@xt.nTuple.decorate(init=xf.init_null)
class Scale_RBFKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_rbf(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

@xt.nTuple.decorate(init=xf.init_null)
class Scale_GaussianKernel(typing.NamedTuple):

    data: xf.Loc

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Scale_Sigmoid, tuple, xf.SiteValue]: ...

    @classmethod
    def f(cls, data):
        return utils.funcs.kernel_gaussian(data)

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert site.loc is not None
        return self.f(self.data.access(state))

# ---------------------------------------------------------------
