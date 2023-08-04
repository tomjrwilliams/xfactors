
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
class BGMM_EM(typing.NamedTuple):
    
    k: int
    sites_data: xt.iTuple

    sites_mu: xt.iTuple
    sites_cov: xt.iTuple
    sites_probs: xt.iTuple

    random: bool = 0.1

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state, small = 10 ** -4):
        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

        data = xf.concatenate_sites(self.sites_data, state)
        mu = xf.concatenate_sites(self.sites_mu, state)
        cov = xf.concatenate_sites(self.sites_cov, state)
        probs = xf.concatenate_sites(self.sites_probs, state)

        # probs = probs + small
        # probs = probs / probs.sum()

        cov = jax.numpy.matmul(
            jax.numpy.transpose(cov, (0, 2, 1)),
            cov,
        )
        # cov = jax.numpy.stack([
        #     jax.numpy.eye(mu.shape[1])
        #     for _ in range(mu.shape[0])
        # ])
        
        if self.random:
            key = xf.get_location(
                self.loc.as_random(), state
            )
            noise = ((
                jax.random.normal(key, shape=cov.shape[:-1])
            ) * self.random)
            diag_noise = jax.numpy.multiply(
                xf.expand_dims(
                    jax.numpy.eye(cov.shape[-1]),
                    0, 
                    noise.shape[0]
                ),
                xf.expand_dims(noise, 1, 1), 
            )
            cov = cov + jax.numpy.abs(diag_noise)

        X = data

        mu_ = xf.expand_dims(mu, 1, data.shape[0])
        data_ = xf.expand_dims(data, 0, mu.shape[0])
        cov_ = xf.expand_dims(
            jax.numpy.linalg.inv(cov), 1, data.shape[0]
        )

        mu_diff = xf.expand_dims(
            jax.numpy.subtract(data_, mu_),
            2, 
            1
        )
        mu_diff_T = jax.numpy.transpose(
            mu_diff,
            (0, 1, 3, 2),
        )

        det = jax.numpy.linalg.det(
            cov,
            #  axis1=1, axis2=2
        )

        norm = 1 / (
            jax.numpy.sqrt(det) * (
                (2 * numpy.pi) ** (data.shape[1] / 2)
            )
        )

        w_unnorm = jax.numpy.exp(
            -(1/2) * (
                jax.numpy.matmul(
                    jax.numpy.matmul(
                        mu_diff,
                        cov_
                    ),
                    mu_diff_T,
                )
            )
        ).squeeze().squeeze().T
        # n_data, n_clusters (?)

        w = jax.numpy.multiply(
            w_unnorm,
            xf.expand_dims(norm, 0, data.shape[0])
        )

        # w_scale = w
        w_scale = jax.numpy.multiply(
            w, xf.expand_dims(probs, 0, w.shape[0])
        )

        w = jax.numpy.divide(
            w_scale,
            xf.expand_dims(
               w_scale.sum(axis=1), 1, w.shape[1]
            )
        )
        # [0 .. 1] proportional score of belonging to each class

        log_likelihood = jax.numpy.log(w_scale.sum(axis=1)).mean()

        data_probs = w
        cluster_weights = w.sum(axis=0)
        cluster_prob = cluster_weights / w.shape[0]

        # return  - (1 / w.shape[1])
        return log_likelihood, cluster_prob

        # w = jax.numpy.divide(
        #     w,
        #     xf.expand_dims(w.sum(axis=0), 0, w.shape[0])
        # )
        # # w = now responsibility

        # w_data = xf.expand_dims(w, -1, mu.shape[1])
        # # n_data, n_cluster, n_col

        # data_ = jax.numpy.transpose(data_, (1, 0, 2))

        # mu_new = jax.numpy.multiply(w_data, data_).sum(axis=0)
        # # jax.numpy.divide(
        # #     ,
        # #     # xf.expand_dims(
        # #     #     denom, 1, mu.shape[1]
        # #     # )
        # # )

        # w_data = jax.numpy.transpose(w_data, (1, 0, 2))
        # w_data = xf.expand_dims(w_data, -1, mu.shape[1])

        # cov_num = jax.numpy.multiply(
        #     w_data,
        #     jax.numpy.matmul(
        #         mu_diff_T, mu_diff, 
        #     ),
        # ).sum(axis=1)
        
        # # cov_new = jax.numpy.divide(
        # #     cov_num,
        # #     # xf.expand_dims(
        # #     #     xf.expand_dims(
        # #     #         denom, -1, cov.shape[-1]
        # #     #     ),
        # #     #     -1, cov.shape[-1]
        # #     # )
        # # )
        # cov_new = cov_num

        # return mu_new, cov_new, data_probs, cluster_prob, log_likelihood

# ---------------------------------------------------------------
