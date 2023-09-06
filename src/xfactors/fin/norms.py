
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

from . import stmts

# ---------------------------------------------------------------

# TODO:

# presumably both from and to periods here

# path of geometric revenue growth rate

# paths of arithmetic (absolute) change in ratios of the above

# ---------------------------------------------------------------

if_none = xf.utils.funcs.if_none
if_none_lazy = xf.utils.funcs.if_none_lazy

OptFloat = typing.Optional[float]

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Norms(typing.NamedTuple):

    in_revenue: OptFloat = 0.
    out_revenue: OptFloat = 0. # ratio on revenue
    # = gross profit

    out_operating: OptFloat = 0. # ratio on revenue
    # = operating profit

    out_investment: OptFloat = 0. # ratio on revenue

    in_interest: OptFloat = 0. # zero when not a bank (?)
    out_interest: OptFloat = 0. # ratio on net debt?

    level_netdebt: OptFloat = 0. # multiple on revenue
    in_netdebt: OptFloat = 0.
    # calculated indirectly via change in net debt

    # residual term
    out_equity: OptFloat = 0.

# ---------------------------------------------------------------
