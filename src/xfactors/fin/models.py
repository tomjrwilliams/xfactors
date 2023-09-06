
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
