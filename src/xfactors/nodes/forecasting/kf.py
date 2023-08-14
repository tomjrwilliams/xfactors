
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

from ... import utils
from ... import xfactors as xf

# ---------------------------------------------------------------

# F = state transition
# H = observation model

# Q = covariance (true) state process noise
# R = covariance of observation noise

# (optional)
# u = control input vector
# B = control-input 

# w = process noise
# w = N(0, Q)

# x = true state
# x = Fx + Bu + w

# v = observation noise
# v = N(0, R)

# z = observation
# z = Hx + v

# ---------------------------------------------------------------

# if non linear function, use extended:

# x = f(x) ... 
# z = h(x) ... 

# where these are differentiable functions
# use jacobian instead of gradient to update covariance

# unscented:

# batch wise sample around current mean instead of point estimate

# given extended assumes we can linearise the function around 
# the current estimates

# ---------------------------------------------------------------

# eg. state = loc and velocity of factors in pca model

# loc = prev(loc) + prev(velocity) + noise
# velocity = velocity + noise

# possibly velocity is mean reverting w.r.t. loc

# then observation is in return space
# via summing up loc * factor weights over tickers

# ---------------------------------------------------------------

# NOTE: parametrise out the state time series
# and then markov assumption, fold into pairs
# and gradient descent mse fit params and state

# optional rand kwargs for random sampling around state

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KF_State_Predicted(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_State_Predicted, tuple, tuple]: ...
    
    # Fx + Bu
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class KF_Cov_Predicted(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_Cov_Predicted, tuple, tuple]: ...
    
    # FPFt + Q
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KF_State_Innovation(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_State_Innovation, tuple, tuple]: ...
    
    # z - Hx
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class KF_Cov_Innovation(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_Cov_Innovation, tuple, tuple]: ...
    
    # HPHt + R
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KF_Kalman_Gain(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_Kalman_Gain, tuple, tuple]: ...
    
    # PHtS-1
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class KF_State_Updated(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_State_Updated, tuple, tuple]: ...
    
    # x + Ky
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class KF_Cov_Updated(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_Cov_Updated, tuple, tuple]: ...
    
    # (I - KH)P
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self


@xt.nTuple.decorate(init=xf.init_null)
class KF_Residual(typing.NamedTuple):
    
    

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[KF_Residual, tuple, tuple]: ...
    
    # z - Hx
    def apply(
        self,
        site: xf.Site,
        state: xf.State
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert False, self

# ---------------------------------------------------------------
