
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


@xt.nTuple.decorate()
class Guard(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

    def apply(self, site, state):
        flags = state[-1]
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()
        
@xt.nTuple.decorate()
class Train(typing.NamedTuple):
    
    flags: dict
    node: xf.Node

    # return tuple of values vmapped over indices
    # given by the values in the map(get_location(site_keys))

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

    def apply(self, site, state):
        flags = state[-1]
        flags["score"] = True
        if all([flags[k] == v for k, v in self.flags.items()]):
            return self.node.apply(site, state)
        return ()

# ---------------------------------------------------------------
