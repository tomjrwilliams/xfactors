
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

from ... import xfactors as xf

from .. import stmts
from .. import norms

from . import defs

from . import kaggle
from . import sec
from . import finnhub
from . import bbg

# ---------------------------------------------------------------

# ---------------------------------------------------------------
