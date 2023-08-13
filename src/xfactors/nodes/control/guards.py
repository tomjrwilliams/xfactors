
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


# TODO: static? or that's a site thing at add stage?

@xt.nTuple.decorate()
class Guard(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Guard, tuple, tuple]: ...
    
    def apply(self, site, state):
        flags = state[-1]
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()
        
        
# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Random(typing.NamedTuple):
    
    node: xf.Node

    random: bool = True

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Train, tuple, tuple]: ...
    
    def apply(self, site, state):
        return self.node.apply(site, state)

# train random, test random etc. to add flagss

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Train(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Train, tuple, tuple]: ...
    
    def apply(self, site, state):
        flags = state[-1]
        flags["train"] = True
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()

@xt.nTuple.decorate()
class Apply(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Apply, tuple, tuple]: ...
    
    def apply(self, site, state):
        flags = state[-1]
        flags["apply"] = True
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()
        
@xt.nTuple.decorate()
class Score(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[Score, tuple, tuple]: ...
    
    def apply(self, site, state):
        flags = state[-1]
        flags["score"] = True
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()

# ---------------------------------------------------------------
