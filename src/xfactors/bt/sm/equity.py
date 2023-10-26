
from __future__ import annotations

import abc
import operator
import collections
import functools
import itertools

import asyncio

import typing
import datetime

import numpy
import pandas # type: ignore

import jax # type: ignore
import jax.numpy # type: ignore
import jax.numpy.linalg # type: ignore

import jaxopt # type: ignore
import optax # type: ignore

import xtuples as xt
import xsm

# ------------------------------------------------------

from . import strats

# ------------------------------------------------------

@xt.nTuple.decorate()
class Ticker(typing.NamedTuple):
    s: str

# ------------------------------------------------------

@xt.nTuple.decorate()
class Allocation(typing.NamedTuple):

    ticker: Ticker
    strategy: strats.Strategy

    # NOTE: strategy not book
    # as can have recursive strategies
    # book is the level at which we flatten down to actually make trades
    # but can be multi-level bindings

    curr: float
    prev: typing.Optional[float] = None

    persist: bool = True

    # --

    @classmethod
    def dependencies(cls):
        return xt.iTuple(())
    
    def matches(self, event: xsm.Event_Like) -> bool:
        return False

    def handler(self, event: xsm.Event_Like):
        return

# ------------------------------------------------------

@xt.nTuple.decorate()
class Order(typing.NamedTuple):

    ticker: Ticker

    curr: float
    prev: typing.Optional[float] = None

    persist: bool = True

    # --

    @classmethod
    def dependencies(cls):
        return xt.iTuple(())
    
    def matches(self, event: xsm.Event_Like) -> bool:
        return False

    def handler(self, event: xsm.Event_Like):
        return
        
# ------------------------------------------------------

@xt.nTuple.decorate()
class Fill(typing.NamedTuple):

    ticker: Ticker

    curr: float
    prev: typing.Optional[float] = None

    persist: bool = True

    # --

    @classmethod
    def dependencies(cls):
        return xt.iTuple(())
    
    def matches(self, event: xsm.Event_Like) -> bool:
        return False

    def handler(self, event: xsm.Event_Like):
        return
        
# ------------------------------------------------------

@xt.nTuple.decorate()
class Position(typing.NamedTuple):

    ticker: Ticker

    curr: float
    prev: typing.Optional[float] = None

    persist: bool = True

    # --

    @classmethod
    def dependencies(cls):
        return xt.iTuple(())
    
    def matches(self, event: xsm.Event_Like) -> bool:
        return False

    def handler(self, event: xsm.Event_Like):
        return

# ------------------------------------------------------
