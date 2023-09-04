
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

# ---------------------------------------------------------------

OptFloat = typing.Optional[float]

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
class Financials_Raw_Data(typing.NamedTuple):

    in_revenue: OptFloat = None
    out_revenue: OptFloat = None
    # = gross profit

    out_operating: OptFloat = None
    # = operating profit

    out_investment: OptFloat = None

    in_interest: OptFloat = None # zero when not a bank (?)
    out_interest: OptFloat = None

    level_netdebt: OptFloat = None
    in_netdebt: OptFloat = None
    # calculated indirectly via change in net debt

    # residual term after above (?)
    # no need for separate terms - ie. allow for negative out_equity
    # where balancing requires an equity capital infusion?
    # just as allow for negative in_netdebt
    out_equity: OptFloat = None

@xt.nTuple.decorate()
class Financials_Raw(typing.NamedTuple):

    source: Source
    identifier: Identifier
    period: Period

    data: Financials_Raw_Data

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Financials_Normalised_Data(typing.NamedTuple):

    in_revenue: OptFloat = None
    out_revenue: OptFloat = None # ratio on revenue
    # = gross profit

    out_operating: OptFloat = None # ratio on revenue
    # = operating profit

    out_investment: OptFloat = None # ratio on revenue

    in_interest: OptFloat = None # zero when not a bank (?)
    out_interest: OptFloat = None # ratio on net debt?

    level_netdebt: OptFloat = None # multiple on revenue
    in_netdebt: OptFloat = None
    # calculated indirectly via change in net debt

    # residual term
    out_equity: OptFloat = None

@xt.nTuple.decorate()
class Financials_Normalised(typing.NamedTuple):

    source: Source
    identifier: Identifier
    period: Period

    data: Financials_Normalised_Data

# ---------------------------------------------------------------

# TODO:

# presumably both from and to periods here

# path of geometric revenue growth rate

# paths of arithmetic (absolute) change in ratios of the above

# ---------------------------------------------------------------

# TODO:

# cluster the above -> 'business models'

# both by value range / walk and volatility thereof (second order stats on the paths)

# including not just single item second order (variance)

# but cross-term: ie. varying covariance per series of particular items

# eg. cluster by how correlated rev is with interest etc.

# NOTE: one way to do the above

# is fit a covariance matrix to each co' path history

# then fit a kernel function over a latent space

# defined over latent embeddings of both the companies

# and the metrics themselves

# interesting to also add embeddings over the periods

# ie. including a latent factor model over time

# but not a full GP - rather we do something like MSE against the 
# per co path cov itself

# so we don't fit against cross-co line item cov as training data

# but our kernel post training can generate such predictions
# which would then be an interesting way to test

# ---------------------------------------------------------------

# TODO:

# covariance of the paths above

# gives a kind of industry structure (particularly if combined / prior on market traded correlations)

# ---------------------------------------------------------------

# TODO: 

# forward extrapolate given parametrisation of the covariance structure

# and walk params for a given name

# to get a forward equity valuation estimate

# or, back out such a parametrisation from the path so far, and the observed equity valuation (and other market prices)

# TODO:

# jointly constrain the walks back from a vector of such prices

# given the observed market correlation structure and the previous financials walk correlations

# ie. say cost(x) correlated rev(y) and rev(y) not being bid up in price terms, expect cost(x) to be subdued (etc.)

# ---------------------------------------------------------------

def read_sec_kaggle():
    return

def parse_sec_kaggle():
    return


def read_sec():
    return

def parse_sec():
    return

# ---------------------------------------------------------------
