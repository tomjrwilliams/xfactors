
from __future__ import annotations

import operator
import collections
# import collections.abc
import functools
import itertools

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import distrax
import jaxopt
import optax

import xtuples as xt

from ... import utils
from ... import xfactors as xf

from . import vanilla

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PCA_Rolling(typing.NamedTuple):
    
    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PCA_Rolling_Encoder(typing.NamedTuple):
    
    n: int
    data: xf.Location
    weights: xf.OptionalLocation = None

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling_Encoder, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        weights = self.weights.access(state)
        data = self.data.access(state)
        return data.map(
            jax.numpy.matmul,
            weights,
            #
        )



@xt.nTuple.decorate(init=xf.init_null)
class PCA_Rolling_Decoder(typing.NamedTuple):
    
    factors: xf.Location
    weights: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling_Decoder, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = xt.ituple(self.weights.access(state))
        factors = self.factors.access(state).map(lambda nd: nd.T)
        return weights.map(jax.numpy.matmul, factors).map(
            lambda nd: nd.T
        )


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Rolling_NegLikelihood(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    # site_encoder: xf.Location

    cov: xf.Location

    # ---

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    random: float = 0

    # todo put the calc into a class method
    # so can be re-used in rolling

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PPCA_Rolling_NegLikelihood, tuple, tuple]: ...

    def apply(self, state, small = 10 ** -4):
        return

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Rolling_EM(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location

    cov: xf.Location

    # ---

    random: float = 0

    # NOTE: we just need covariance here

    # so we can pass this from a kernel function
    # if we want

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PPCA_Rolling_EM, tuple, tuple]: ...
    
    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        # use noisy_sgd instead of random
        # if self.random:
        #     key = xf.get_location(
        #         self.loc.as_random(), state
        #     )
        #     weights = weights + ((
        #         jax.random.normal(key, shape=weights.shape)
        #     ) * self.random)

        # feature * feature
        S = cov
        d = weights.shape[0] # n_features

        W = weights

        mm = jax.numpy.matmul

        noise = jax.numpy.eye(weights.shape[1]) * sigma_sq
        M = mm(W.T, W) + noise

        invM = jax.numpy.linalg.inv(M)

        # M = factor by factor
        # C = feature by feature

        SW = mm(S, W)

        W_new = mm(SW, jax.numpy.linalg.inv(
            noise + mm(mm(invM, W.T), SW)
        ))
        
        sigma_sq_new = (1 / d) * jax.numpy.trace(
            jax.numpy.subtract(S, mm(
                SW, mm(invM, W_new.T)
            ))
        )

        return W_new, jax.numpy.sqrt(sigma_sq_new + small)

# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Rolling_Marginal_Observations(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    encoder: xf.Location
    data: xf.Location

    cov: xf.Location

    # ---

    random: float = 0

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PPCA_Rolling_Marginal_Observations, tuple, tuple]: ...
    
    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = xf.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features

        noise = sigma_sq * jax.numpy.eye(weights.shape[0])

        C = jax.numpy.matmul(weights, weights.T) + noise
        mu = jax.numpy.zeros(weights.shape[0])

        dist = distrax.MultivariateNormalFullCovariance(mu, C)

        return dist.log_prob(data)


@xt.nTuple.decorate(init=xf.init_null)
class PPCA_Rolling_Conditional_Latents(typing.NamedTuple):
    
    sigma: xf.Location
    weights: xf.Location
    encoder: xf.Location
    data: xf.Location

    cov: xf.Location

    # ---
   
    random: float = 0

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PPCA_Rolling_Conditional_Latents, tuple, tuple]: ...
    
    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = self.sigma.access(state)
        sigma_sq = jax.numpy.square(sigma)

        weights = self.weights.access(state)
        cov = self.cov.access(state) # of obs

        data = self.data.access(state)
    
        # N = xf.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features
        factors = self.encoder.access(state)
        # replace with factors
        
        mu = jax.numpy.zeros(weights.shape[0]) # obs mu

        noise = jax.numpy.eye(weights.shape[1]) * sigma_sq

        mm = jax.numpy.matmul

        W = weights
        M = mm(weights.T, weights) + noise
        M_inv = jax.numpy.linalg.inv(M)

        # will need to expand mu to match shape of data
        dist = distrax.MultivariateNormalFullCovariance(
            mm(mm(M_inv, W.T), data - mu),
            sigma_sq * M_inv
        )
        return dist.log_prob(factors)

# ---------------------------------------------------------------
