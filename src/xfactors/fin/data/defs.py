
from __future__ import annotations

import typing
import collections
import functools
import itertools
import operator

import datetime

import numpy
import pandas

import jax
import jaxopt
import optax

import xtuples as xt

from .. import xfactors as xf

from .. import stmts
from .. import norms

# ---------------------------------------------------------------

FULL = -1

@xt.nTuple.decorate()
class Source(typing.NamedTuple):
    name: str
    detail: typing.Optional[int] = None

    # and release information?

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Identifier(typing.NamedTuple):
    id: str

    # include release information as well?
    # or that's perhaps more a source question?

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Period(typing.NamedTuple):
    year: int

    # quarter?

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class Statements(typing.NamedTuple):

    source: Source
    identifier: Identifier
    period: Period

    data: stmts.Statements


@xt.nTuple.decorate()
class Norms(typing.NamedTuple):

    source: Source
    identifier: Identifier
    period: Period

    data: norms.Norms

# ---------------------------------------------------------------
