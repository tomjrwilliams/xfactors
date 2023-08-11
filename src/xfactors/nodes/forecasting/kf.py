
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


@xt.nTuple.decorate()
class KF_State_Predicted(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # Fx + Bu
    def apply(self, state):
        assert False, self


@xt.nTuple.decorate()
class KF_Cov_Predicted(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # FPFt + Q
    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class KF_State_Innovation(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # z - Hx
    def apply(self, state):
        assert False, self


@xt.nTuple.decorate()
class KF_Cov_Innovation(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # HPHt + R
    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class KF_Kalman_Gain(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # PHtS-1
    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class KF_State_Updated(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # x + Ky
    def apply(self, state):
        assert False, self


@xt.nTuple.decorate()
class KF_Cov_Updated(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # (I - KH)P
    def apply(self, state):
        assert False, self


@xt.nTuple.decorate()
class KF_Residual(typing.NamedTuple):
    
    sites: xt.iTuple

    

    # z - Hx
    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------
