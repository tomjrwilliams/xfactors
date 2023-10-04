
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
from . import norms
from . import data

# ---------------------------------------------------------------

# NOTE: modelling process

# pull data as stmt history
# map to norm

# take sub-set of the norm fields

# iterate one step forward
# either walk of revenue geometric growth
# or arithmetic walk of the other ratios

# use next(norm) to generate curr_stmt, given prev_stmt

# minimise error
# under appropriate constraints

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Factor_Temporal(typing.NamedTuple):

    # any factor with an embedding per time period

    # specifying a group of fields
    # presumably also including market data derivatives 
    # including to specific other reference assets

    # that are multiplied through by that particular embedding

    # perhaps even derivatives of fields
    # eg. rolling corr of change in field to a given price

    # idea here is to separate out different market / sector
    # exposures
    # independent of the business model (below)

    pass
    
# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Factor_Discriminatory(typing.NamedTuple):

    # factors with embeddings that are in some way constant (?)
    # independent of time for the name in question

    # ie. inventory / fixed assets share of assets / revenue

    # idea with this is to separate out different business models
    # independent of the market / sector exposures (above)

    pass

# where the superficial issue with the above is that the business model can change with time
# but the above is not 'independent' of time
# in the sense that we wouldn't get a different loading per period

# but only in that the *embedding*
# the loadings are multiplied through by
# doesn't change

# ---------------------------------------------------------------

# NOTE: fields 

# or derivatives thereof (eg. rolling corr of changes with a given market price)
# or relevant market price derivatives for the name in question

# grouped into either of the two factors above

# then per period multiplied through by either 
# the relevant factor embedding for the period (for the #1)

# or the discriminatory embedding (s)? if such a thing

# to get a combined vector of factor exposure values
# and business model discrimination values

# that are then passed into a final model
# mapping to a stmt spanning set of norm fields

# where spanning means that we can derive a full stmt
# given a prev stmt and the fields

# even if it means some operation in stmt space
# eg. summing up / subbing through

# where any fields in the stmt not mapped to
# will thus be ignored
# and end up in the relevant residual

# ---------------------------------------------------------------

# minimum spanning fields

# - revenue
# direct

# - assets
# could be direct to revenue, but probably better
# to model each non-ignored item separately, then sum up at stmt level

# - liab
# as per assets

# IS:
# - either ignored, derived
# or modelled from revenue

# CF:
# - either ignored
# - investing: from revenue (capex, intangibles)
# - financing: from revenue (change in assets / liab items)
# - operating: mostly flow through from the relevant IS items 

# - equity
# residual from the rest of the stmts

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

# NOTE: in terms of strategy implementation

# the point here is second order effects

# ie. given a particular thesis, more specifically
# a forecast for some variable

# say, sales of some product category
# the direct beneficiary might already have that priced in

# or might have low operating leverage to the outcome

# whereas second order relationships might either
# not yet be priced in

# or have a greater sensitivity, even if one step removed

# ---------------------------------------------------------------
