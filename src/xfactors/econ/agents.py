from __future__ import annotations

import typing

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

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Representation(typing.NamedTuple):
    pass


@xt.nTuple.decorate()
class Agent(typing.NamedTuple):

    # TODO: agents have:

    # - scopes 
    # defining what state they have access to
    # with what noise, lag, etc.

    # - decision targets
    # defining what choices the agents make each period
    # either discrete or numeric

    # - representations
    # representing what the agent currently believes
    # has measured / estimated about something

    # - models
    # functions
    # of the exposed value of the agents decision (s)
    # ie. they can observe their own that-period decision with noise
    # and any other exposed state
    # and any previous representations
    # to produce new representations
    # so the optimise takes care of wrapping the decision site into
    # a representation (an exposed state or model function thereof)

    # - optimisation target
    # final representation the agents optimises over

    # such that we can run the jaxopt form:
    # agent.optimise(decisions, state)
    # to set decisions per period, to pass to environment

    pass

# ---------------------------------------------------------------
