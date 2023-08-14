
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

from .... import utils
from .... import xfactors as xf

# ---------------------------------------------------------------

# eg. latent features = equity sectors
# n_latents > factors, zero weights on extras (noise factors)

# so each sector (per latent_factor) has weighting
# on the equivalent index loading factor
# with 1 in the features (tickers) in that sector, zero elsewhere


@xt.nTuple.decorate(init=xf.init_null)
class PCA_Rolling_LatentWeightedMean_MSE(typing.NamedTuple):
    
    # sites
    weights_pca: xf.Location
    weights_structure: xf.Location
    latents: xf.Location

    # assume feature * factor minimum

    share_factors: bool = True
    share_latents: bool = True

    def init(
        self, site: xf.Site, model: xf.Model, data: tuple
    ) -> tuple[PCA_Rolling_LatentWeightedMean_MSE, tuple, tuple]: ...

    def f_apply(
        self, weights_pca, weights_structure, latents, 
    ):
        # weights_pca = features * factors 
        # weights_structure = n_latents * features

        if self.share_factors:
            weights_structure = xf.expand_dims(
                weights_structure, 0, weights_pca.shape[-1]
            )

        if self.share_latents:
            weights_structure = xf.expand_dims(
                weights_structure, 0, latents.shape[0]
            )

        # n_latents, n_factors, n_features, latent_features
        weights_structure = jax.numpy.transpose(
            weights_structure,
            (0, 3, 2, 1),
        )

        weights_pca_agg = jax.numpy.multiply(
            xf.expand_dims(
                xf.expand_dims(weights_pca, 0, 1), 0, 1
            ),
            weights_structure,
        ).sum(axis=-1).sum(axis=-1)
        # n_latents, latent_features, factors, features
        # n_latents, latent_features

        return jax.numpy.square(weights_pca_agg - latents).mean()

    def apply(
        self,
        site: xf.Site,
        state: xf.State,
        model: xf.Model,
    ) -> typing.Union[tuple, jax.numpy.ndarray]:

        latents = self.latents.access(state)

        weights_pca = xt.ituple(self.weights_pca.access(state))
        weights_structure = xt.ituple(self.weights_structure.access(state))

        res = weights_pca.map(
            functools.partial(self.f_apply, latents=latents),
            weights_structure,
        )
        return jax.numpy.vstack(res).mean()

# ---------------------------------------------------------------
