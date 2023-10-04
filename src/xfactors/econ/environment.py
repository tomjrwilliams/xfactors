from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from ... import xfactors as xf
from ... import utils
from ... import visuals

from . import states
from . import agents

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Environment(typing.NamedTuple):

    # TODO: environment has:

    # - state database
    # together with meta-data on what has scope over what
    # so can drop redundant data as needed

    # - agents
    # making decisions

    # - dynamics
    # mapping how the decisions
    # themselves state objects
    # are propagated into the next overall state, given prev state
    # and any stochasticity

    # for instance:
    # market clearing dynamic given orders from agents
    # random technology path
    # random shock to availability of various resources, etc.
    # agent death, and emergence, etc.

    pass

# ---------------------------------------------------------------
