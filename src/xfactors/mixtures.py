
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

import jax.scipy.special

digamma = jax.scipy.special.digamma

# ---------------------------------------------------------------

# from jax.config import config 
# config.update("jax_debug_nans", True) 

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class GMM(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
        data = xf.concatenate_sites(self.sites, state, axis = 1)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class BGMM_Spherical_EM(typing.NamedTuple):
    
    k: int
    sites_data: xt.iTuple

    sites_mu: xt.iTuple
    sites_a: xt.iTuple
    sites_b: xt.iTuple
    sites_probs: xt.iTuple
    site_alpha: xf.Location

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        # https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf
        # https://scikit-learn.org/0.15/modules/dp-derivation.html

        data = xf.concatenate_sites(self.sites_data, state)
        # mu = xf.concatenate_sites(self.sites_mu, state)
        # var = xf.concatenate_sites(self.sites_mu, state)

        X = data
        
        # gamma probs
        a = jax.numpy.exp(
            xf.concatenate_sites(self.sites_a, state)
        )
        b = jax.numpy.exp(
            xf.concatenate_sites(self.sites_b, state)
        )
        # n_clusters

        probs = xf.concatenate_sites(self.sites_probs, state)
        # of size n_points n_clusters

        alpha = jax.numpy.exp(
            xf.get_location(self.site_alpha, state)
        )
        # alpha = 2.

        ba = jax.numpy.divide(b, a)
        ba_expand = xf.expand_dims(ba, 0, data.shape[0])

        prob_cluster_sum = probs.sum(axis=0)

        gamma_1 = 1 + prob_cluster_sum
        gamma_2 = jax.numpy.flip(
            jax.numpy.concatenate([
                jax.numpy.zeros(1),
                jax.numpy.cumsum(
                    jax.numpy.flip(prob_cluster_sum)
                )
            ])[:-1]
            # sum over probs over all data points
            # where index > cluster index
        ) + a
        # + alpha

        ba_probs = jax.numpy.multiply(probs, ba_expand)

        # n_data, n_clusters
        ba_probs_exp = xf.expand_dims(ba_probs, -1, data.shape[1])
        data_exp = xf.expand_dims(data, 1, a.shape[0])
        # n_data, n_clusters, n_cols

        mu_num = jax.numpy.multiply(ba_probs_exp, data_exp).sum(axis=0)
        # n_clusters, n_cols

        mu_den = 1 + ba_probs.sum(axis=0)
        # n_clusters

        mu_new = jax.numpy.divide(
            mu_num, xf.expand_dims(mu_den, -1, mu_num.shape[1])
        )

        # n cols?
        # dims of normal distribution, from which we draw
        # the covar?
        D = mu_num.shape[1]

        # mu new?
        mu_exp = xf.expand_dims(mu_new, 0, data.shape[0])
        # n_data, n_clusters, n_cols

        mu_diff_sq = jax.numpy.square(jax.numpy.subtract(
            data_exp, mu_exp
        )).sum(axis = -1)
        # sum out cols
        # so n_data, n_clusters

        a_new = 1 + ((D / 2) * probs.sum(axis = 0))
        b_new = 1 + jax.numpy.multiply(
            probs, mu_diff_sq + D,
        ).sum(axis=0) / 2

        expand = lambda v: xf.expand_dims(v, 0, data.shape[0])

        E_logPX = (
            - ((D / 2) * jax.numpy.log(2 * numpy.pi))
            + expand(
                (D / 2) * (digamma(a_new) - jax.numpy.log(b_new))
            )
            - jax.numpy.multiply(
                expand(
                    a_new / (2 * b_new)
                ),
                mu_diff_sq + D
            )
            - jax.numpy.log(2 * numpy.pi * numpy.e)
        )
        # size = n_data, n_clusters

        E_logPX_prob = jax.numpy.multiply(
            E_logPX, probs
        )
        # .sum(axis=1)

        probs_new = jax.numpy.exp(
            + expand(digamma(gamma_1)) # vector
            - expand(digamma(gamma_1 + gamma_2)) # vector
            + E_logPX
            + expand(jax.numpy.concatenate([
                jax.numpy.zeros(1),
                jax.numpy.cumsum(
                    digamma(gamma_2) - digamma(gamma_1 + gamma_2)
                )
            ])[:-1]) # vector
        )
        # n_data, n_clusters

        probs_norm = jax.numpy.divide(
            probs_new,
            xf.expand_dims(
                probs_new.sum(axis=1), 1, probs_new.shape[1]
            )
        )
        # probs_norm = jax.nn.softmax(probs_new, axis = 1)

        return (
            mu_new,
            jax.numpy.log(a_new),
            jax.numpy.log(b_new),
            probs_new,
            probs_norm,
            # E_logPX_prob,
        )

        # TODO: try gradient descent brute force maximising
        # e_logPX

# ---------------------------------------------------------------
