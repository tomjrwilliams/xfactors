
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
    sites: xt.iTuple

    

    def init_shape(self, site, model, data):
        objs = self.sites.map(xf.f_get_location(model))
        return self._replace(
            shape = (
                objs.map(lambda o: o.shape[1]).pipe(sum),
                self.n,
            )
        )

    def apply(self, state):
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class PCA_Rolling_Encoder(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple
    site: xf.OptionalLocation = None

    
    

    def init_shape(self, site, model, data):
        objs = self.sites.map(xf.f_get_location(model))
        return self._replace(
            shape = (
                objs.map(lambda o: o.shape[1]).pipe(sum),
                self.n,
            )
        )

    def init_params(self, site, model, data):
        if self.site is None:
            return self._replace(
                site=self.loc.as_param()
            ), utils.rand.gaussian(self.shape)
        # TODO: check below, assumes weights generated elsewhere
        return self, utils.rand.gaussian(self.shape)

    def apply(self, state):
        weights = xf.get_location(self.site, state)
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        return jax.numpy.matmul(data, weights)



@xt.nTuple.decorate()
class PCA_Rolling_Decoder(typing.NamedTuple):
    
    sites: xt.iTuple

    # sites_weight: xt.iTuple
    # sites_data: xt.iTuple

    # TODO: generalise to sites_weight and sites_data
    # so that can spread across multiple prev stages
    # and then concat both, or if size = 1, then as below
    # can also pass as a nested tuple? probs cleaner to have separate

    # todo. split out the apply method
    # to a class method on the non rolling class

    
    

    def apply(self, state):
        assert len(self.sites) == 2
        l_site, r_site = self.sites
        weights = xf.get_location(r_site, state)
        data = xf.get_location(l_site, state)
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
    loadings: xt.iTuple
    weights: xt.iTuple
    latents: xt.iTuple

    def apply(self, state):

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

        loadings = xf.concatenate_sites(self.loadings, state)
        # periods * features * factors 
        loadings = jax.numpy.transpose(loadings, (0, 2, 1))

        weights = xf.concatenate_sites(self.weights, state)
        # n_latents * latent_features * factors * features

        latents = xf.concatenate_sites(self.latents, state)
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
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    # site_encoder: xf.Location

    site_cov: xf.Location

    # ---

    
    

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    random: float = 0

    # todo put the calc into a class method
    # so can be re-used in rolling

    def apply(self, state, small = 10 ** -4):
        return

# ---------------------------------------------------------------


@xt.nTuple.decorate()
class PPCA_Rolling_EM(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    # site_encoder: xf.Location

    site_cov: xf.Location

    # ---

    
    

    random: float = 0

    # NOTE: we just need covariance here

    # so we can pass this from a kernel function
    # if we want

    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = xf.get_location(self.site_sigma, state)
        sigma_sq = jax.numpy.square(sigma)

        weights = xf.concatenate_sites(self.sites_weights, state)
        cov = xf.get_location(self.site_cov, state) # of obs

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
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    site_encoder: xf.Location
    site_date: xf.Location

    site_cov: xf.Location

    # ---

    
    

    random: float = 0

    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = xf.get_location(self.site_sigma, state)
        sigma_sq = jax.numpy.square(sigma)

        weights = xf.concatenate_sites(self.sites_weights, state)
        cov = xf.get_location(self.site_cov, state) # of obs

        data = xf.get_location(self.site_data, state)
    
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
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    site_encoder: xf.Location
    site_date: xf.Location

    site_cov: xf.Location

    # ---

    
    

    random: float = 0

    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = xf.get_location(self.site_sigma, state)
        sigma_sq = jax.numpy.square(sigma)

        weights = xf.concatenate_sites(self.sites_weights, state)
        cov = xf.get_location(self.site_cov, state) # of obs

        data = xf.get_location(self.site_data, state)
    
        # N = xf.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features
        factors = xf.get_location(self.site_encoder, state)
        
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
