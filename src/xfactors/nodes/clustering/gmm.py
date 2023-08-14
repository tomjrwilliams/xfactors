
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

import jaxopt
import optax

import xtuples as xt

from ... import xfactors as xf

# ---------------------------------------------------------------

import jax.scipy.special

digamma = jax.scipy.special.digamma


# ---------------------------------------------------------------


@xt.nTuple.decorate(init=xf.init_null)
class GMM(typing.NamedTuple):
    
    n: int
    data: xf.Location

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[GMM, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model
        data = self.data.access(state)
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

# ---------------------------------------------------------------

small = 10 ** -4

@xt.nTuple.decorate(init=xf.init_null)
class BGMM_EM(typing.NamedTuple):
    
    k: int
    data: xf.Location

    mu: xf.Location
    cov: xf.Location
    probs: xf.Location

    noise: typing.Optional[float] = 0.1

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[BGMM_EM, tuple, tuple]: ...
    
    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        # https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

        data = self.data.access(state)
        mu = self.mu.access(state)
        cov = self.cov.access(state)
        probs = self.probs.access(state)

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
        
        if self.noise:
            assert site.loc is not None
            key = xf.get_location(
                site.loc.as_random(), state
            )
            noise = ((
                jax.random.normal(key, shape=cov.shape[:-1])
            ) * self.noise)
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

        # w_scale = jax.numpy.multiply(
        #     w, xf.expand_dims(cluster_prob, 0, w.shape[0])
        # )

        w_scale = jax.numpy.divide(
            w,
            xf.expand_dims(
               w.sum(axis=1), 1, w.shape[1]
            )
        )
        # [0 .. 1] proportional score of belonging to each class

        log_likelihood = jax.numpy.log(w.sum(axis=1)).mean()

        max_w = w.T[jax.numpy.argmax(w, axis = 1)]
        mean_w = w.sum(axis=1) - max_w

        separability = (max_w - mean_w).mean()

        data_probs = w

        cluster_weights = w_scale.sum(axis=0)
        cluster_prob = cluster_weights / w.shape[0]

        cluster_inv = 1 / (cluster_prob + 0.01)
        cluster_inv = cluster_inv / cluster_inv.sum()
        # this woudl be if averaging over individual points

        r = jax.numpy.divide(
            w,
            xf.expand_dims(
                w.sum(axis=0), 0, w.shape[0]
            )
        )
        # w = now responsibility

        r_data = xf.expand_dims(r, -1, mu.shape[1])
        # n_data, n_cluster, n_col

        data_ = jax.numpy.transpose(data_, (1, 0, 2))

        mu_new = jax.numpy.multiply(r_data, data_).sum(axis=0)

        r_data = jax.numpy.transpose(r_data, (1, 0, 2))
        r_data = xf.expand_dims(r_data, -1, mu.shape[1])

        cov_num = jax.numpy.multiply(
            r_data,
            jax.numpy.matmul(
                mu_diff_T, mu_diff, 
            ),
        ).sum(axis=1)
        
        cov_new = cov_num

        # pivot = jax.numpy.multiply(
        #     mu,
        #     xf.expand_dims(cluster_inv, 1, mu.shape[1])
        # ).sum(axis = 0)
        pivot = data.mean(axis = 0)
        # # sum out clusters

        # residual = jax.numpy.subtract(
        #     mu_new,
        #     xf.expand_dims(pivot, 0, mu_new.shape[0]),
        # )

        return mu_new, cov_new, data_probs, log_likelihood, separability
        # , cluster_prob, log_likelihood

# ---------------------------------------------------------------
