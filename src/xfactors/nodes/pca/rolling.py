
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


@xt.nTuple.decorate()
class PCA_Rolling(typing.NamedTuple):
    
    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        data = self.data.access(state)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------


@xt.nTuple.decorate()
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
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        assert self.weights is not None
        weights = self.weights.access(state)
        data = self.data.access(state)
        return jax.numpy.matmul(data, weights)



@xt.nTuple.decorate()
class PCA_Rolling_Decoder(typing.NamedTuple):
    
    data: xf.Location
    weights: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling_Decoder, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        weights = self.weights.access(state)
        data = self.data.access(state)
        return jax.numpy.matmul(weights, data.T)

# ---------------------------------------------------------------

# eg. latent features = equity sectors
# n_latents > factors, zero weights on extras (noise factors)

# so each sector (per latent_factor) has weighting
# on the equivalent index loading factor
# with 1 in the features (tickers) in that sector, zero elsewhere


@xt.nTuple.decorate()
class PCA_Rolling_LatentWeightedMean_MSE(typing.NamedTuple):
    
    # sites
    loadings: xf.Location
    weights: xf.Location
    latents: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling_LatentWeightedMean_MSE, tuple, tuple]: ...

    def apply(
        self,
        site: xf.Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        # TODO: not concatenate
        # irregular shapes are likely so can't be a single data structure

        # ie. the rolling pca has to return a tuple not a single array
        # makes eg. rolling universe simpler as well if covariance only has the values needed for each period
        
        # but the joining still makes sense
        # so join_sites (tuple join instead of concatenate)
        # can even be a tuple of tuple to still allow concatenation

        # or we drop the idea of site concatenation?
        # and have explicit concatenation operators
        # probably that...

        loadings = self.loadings.access(state)
        # periods * features * factors 
        loadings = jax.numpy.transpose(loadings, (0, 2, 1))

        weights = self.weights.access(state)
        # n_latents * latent_features * factors * features

        latents = self.latents.access(state)
        # n_latents * latent_features

        weighted_loadings = jax.numpy.multiply(
            xf.expand_dims(
                xf.expand_dims(loadings, 0, 1), 0, 1
            ),
            xf.expand_dims(weights, 2, 1),
        ).sum(axis=-1).sum(axis=-1)
        # n_latents, latent_features, periods, factors, features
        # n_latents, latent_features, periods

        weighted_loadings = jax.numpy.transpose(
            weighted_loadings, (2, 0, 1,)
        )

        return jax.numpy.square(
            xf.expand_dims(latents, 0, 1),
            weighted_loadings,
        ).mean()

# ---------------------------------------------------------------


@xt.nTuple.decorate()
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


@xt.nTuple.decorate()
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


@xt.nTuple.decorate()
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


@xt.nTuple.decorate()
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
