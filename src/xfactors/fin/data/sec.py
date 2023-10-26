
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

# https://www.kaggle.com/datasets/securities-exchange-commission/financial-statement-extracts

# https://www.sec.gov/dera/data/financial-statement-data-sets

# ---------------------------------------------------------------
