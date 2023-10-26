
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

# ---------------------------------------------------------------

# https://www.kaggle.com/datasets/finnhub/reported-financials

# follows: https://finnhub.io/docs/api#financials-reported

# ---------------------------------------------------------------
