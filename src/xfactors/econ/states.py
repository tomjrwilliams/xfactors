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

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class State(typing.NamedTuple):

    # TODO: states have:

    # - values

    # - accessibility tags
    # determining which agents (if any)
    # have visibility on the value of the state
    # under what delay, noise, etc.
    # given their respective scopes

    pass

# ---------------------------------------------------------------
