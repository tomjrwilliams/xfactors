
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

from . import rand
from . import dates
from . import xfactors as xf

# ---------------------------------------------------------------

# from jax.config import config 
# config.update("jax_debug_nans", True) 

def calc_loadings(eigvals, eigvecs):
    return jax.numpy.multiply(
        # jax.numpy.resize(
        #     jax.numpy.expand_dims(eigvals, 0),
        #     eigvecs.shape,
        # ),
        xf.expand_dims_like(eigvals, axis=0, like=eigvecs),
        eigvecs,
    )

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class PCA(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
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

@xf.operator_bindings()
@xt.nTuple.decorate
class PCA_Encoder(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple
    site: xf.Location = None

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

    def init_shape(self, model, data):
        objs = self.sites.map(xf.f_get_location(model))
        return self._replace(
            shape = (
                objs.map(lambda o: o.shape[1]).pipe(sum),
                self.n,
            )
        )

    def init_params(self, model, state):
        if self.site is None:
            return self._replace(
                site=self.loc.as_param()
            ), rand.gaussian(self.shape)
        # TODO: check below, assumes weights generated elsewhere
        return self, rand.gaussian(self.shape)

    def apply(self, state):
        weights = xf.get_location(self.site, state)
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        return jax.numpy.matmul(data, weights)


@xf.operator_bindings()
@xt.nTuple.decorate
class PCA_Decoder(typing.NamedTuple):
    
    sites: xt.iTuple

    # sites_weight: xt.iTuple
    # sites_data: xt.iTuple

    # TODO: generalise to sites_weight and sites_data
    # so that can spread across multiple prev stages
    # and then concat both, or if size = 1, then as below
    # can also pass as a nested tuple? probs cleaner to have separate

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

    def apply(self, state):
        assert len(self.sites) == 2
        l_site, r_site = self.sites
        weights = xf.get_location(r_site, state)
        data = xf.get_location(l_site, state)
        return jax.numpy.matmul(weights, data.T)

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_NegLikelihood(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    # site_encoder: xf.Location

    site_cov: xf.Location

    # ---

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    random: float = 0

    # todo put the calc into a class method
    # so can be re-used in rolling

    def apply(self, state, small = 10 ** -4):
        # https://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf

        sigma = xf.get_location(self.site_sigma, state)
        sigma_sq = jax.numpy.square(sigma)

        weights = xf.concatenate_sites(self.sites_weights, state)
        cov = xf.get_location(self.site_cov, state) # of obs
    
        # N = xf.get_location(self.site_encoder, state).shape[0]

        # feature * feature
        S = cov
        # d = weights.shape[0] # n_features

        if self.random:
            key = xf.get_location(
                self.loc.as_random(), state
            )
            weights = weights + ((
                jax.random.normal(key, shape=weights.shape)
            ) * self.random)
    
        W = weights
        
        noise = jax.numpy.eye(weights.shape[0]) * sigma_sq
        C = jax.numpy.add(
            jax.numpy.matmul(weights, weights.T),
            noise
        )
        # noise = jax.numpy.eye(weights.shape[1]) * sigma_sq
        # M = jax.numpy.add(
        #     jax.numpy.matmul(weights.T, weights),
        #     noise
        # )
        # invM = jax.numpy.linalg.inv(M)

        detC = jax.numpy.linalg.det(C)
        invC = jax.numpy.linalg.inv(C)

        invC_S = jax.numpy.matmul(invC, S)
        trace_invC_S = jax.numpy.trace(invC_S)

        # max this (dropped the * N)
        L = - (
            # + (d * jax.numpy.log(2 * numpy.pi))
            + jax.numpy.log(detC) + trace_invC_S
        ) / 2
        # so min neg(L)

        return -L
        
# ---------------------------------------------------------------

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_Rolling_NegLikelihood(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    # site_encoder: xf.Location

    site_cov: xf.Location

    # ---

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

    # NOTE: direct minimisation with gradient descent
    # doesn't seem to recover pca weights

    random: float = 0

    # todo put the calc into a class method
    # so can be re-used in rolling

    def apply(self, state, small = 10 ** -4):
        return

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_EM(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    # site_encoder: xf.Location

    site_cov: xf.Location

    # ---

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

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

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_Marginal_Observations(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    site_encoder: xf.Location
    site_date: xf.Location

    site_cov: xf.Location

    # ---

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

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

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_Conditional_Latents(typing.NamedTuple):
    
    site_sigma: xf.Location
    sites_weights: xt.iTuple
    site_encoder: xf.Location
    site_date: xf.Location

    site_cov: xf.Location

    # ---

    loc: xf.Location = None
    shape: xt.iTuple = None
    train: bool = None

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
