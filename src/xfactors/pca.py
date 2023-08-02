
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
        data = jax.numpy.concatenate(
            self.sites.map(xf.f_get_location(state)),
            axis=1,
        )
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

@xf.operator_bindings()
@xt.nTuple.decorate
class PCA_Encoder(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple
    site: xf.Location = None

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

    def init_params(self, model, state):
        if self.site is None:
            return self._replace(
                site=self.loc.as_param()
            ), rand.gaussian(self.shape)
        # TODO: check below, assumes weights generated elsewhere
        return self, rand.gaussian(self.shape)

    def apply(self, state):
        weights = xf.get_location(self.site, state)
        return jax.numpy.matmul(
            jax.numpy.concatenate(
                self.sites.map(xf.f_get_location(state)),
                axis=1,
            ),
            weights,
        )


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

    def apply(self, state):
        assert len(self.sites) == 2
        l_site, r_site = self.sites
        weights = l_site
        return jax.numpy.matmul(
            xf.get_location(r_site, state), 
            xf.get_location(l_site, state).T
        )

@xf.operator_bindings()
@xt.nTuple.decorate
class VMap_PCA_Decoder(typing.NamedTuple):
    
    sites_weight: xt.iTuple
    sites_data: xt.iTuple

    # the sites are not per vmap entry
    # but rather concatenated, and then vmapped over

    # so each site should return a tuple to vmap over
    # tuple because can then be irregular size

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False

# ---------------------------------------------------------------

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_EM_E(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

@xf.operator_bindings()
@xt.nTuple.decorate
class PPCA_EM_M(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: xf.Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert False, self

# ---------------------------------------------------------------
