
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

from . import funcs

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Constraint_EM(typing.NamedTuple):
    
    param: xf.Location
    optimal: xf.Location # optimal at this step from em algo

    cut_tree: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_EM, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        param = self.param.access(state)
        optimal = self.optimal.access(state)
        return funcs.loss_mse(
            param,
            ( 
                jax.lax.stop_gradient(optimal)
                if self.cut_tree
                else optimal
            )
        )


@xt.nTuple.decorate()
class Constraint_EM_MatMul(typing.NamedTuple):
    
    raw: xf.Location
    optimal: xf.Location # optimal at this step from em algo

    cut_tree: bool = False

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Constraint_EM_MatMul, tuple, tuple]: ...
    
    def apply(self, site: xf.Site, state: tuple) -> tuple:
        raw = self.raw.access(state)
        optimal = self.optimal.access(state)
        param = jax.numpy.matmul(
            jax.numpy.transpose(raw, (0, 2, 1)),
            raw,
        )
        return funcs.loss_mse(
            param,
            ( 
                jax.lax.stop_gradient(optimal)
                if self.cut_tree
                else optimal
            )
        )

# ---------------------------------------------------------------

