
import itertools
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf

from tests import utils

import jax.config
jax.config.update("jax_debug_nans", True)

from sklearn.cluster import KMeans

def test_lgp() -> bool:

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    N_COLS = 5
    N_CLUSTERS = 3
    N_FACTORS = 5
    N_VARIABLES = N_CLUSTERS * N_COLS

    VARIABLES = xt.iTuple.range(N_VARIABLES)
    CLUSTERS = xt.iTuple.range(N_CLUSTERS)
    FACTORS  = xt.iTuple.range(N_FACTORS)

    CLUSTER_MEMBERS = {
        i: chunk
        for i, chunk 
        in VARIABLES.chunkby(lambda i: i // N_COLS).enumerate()
    }
    CLUSTER_MAP = xt.iTuple(CLUSTER_MEMBERS.items()).fold(
        lambda acc, cluster_chunk: {
            **acc,
            **{
                i: cluster_chunk[0]
                for i in cluster_chunk[1]
            }
        },
        initial={},
    )

    # mu = xf.utils.rand.orthogonal(N_FACTORS)[..., :N_CLUSTERS]
    mu = xf.utils.rand.gaussian((N_FACTORS, N_CLUSTERS,))
    
    betas = numpy.add(
        numpy.array([
            [
                mu[f][CLUSTER_MAP[i]]
                for i in VARIABLES
            ]
            for f in FACTORS
        ]),
        xf.utils.rand.gaussian((N_FACTORS, N_VARIABLES,)) / 10
    ).T
    cov = numpy.matmul(
        numpy.matmul(betas, numpy.eye(N_FACTORS)), betas.T
    )
    cov = numpy.divide(
        cov, 
        xf.expand_dims_like(
            cov.sum(axis=1), axis=1, like=cov
        ),
        # numpy.resize(
        #     numpy.expand_dims(cov.sum(axis=1), axis=1),
        #     cov.shape,
        # )
    )

    vs = xf.utils.rand.v_mv_gaussian(
        100,
        mu=numpy.zeros((N_VARIABLES,)), 
        cov=cov
    )
    assert not numpy.isnan(vs).any(), betas

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    
    model, STAGES = xf.Model().init_stages(3)
    INPUT, COV, LATENT, GP = STAGES

    model = (
        model.add_input(xf.nodes.inputs.dfs.Input_DataFrame_Wide())
        .add_node(COV, xf.nodes.cov.vanilla.Cov(
            data=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ), static=True,
        ))
        .add_node(LATENT, xf.latents.Latent(
            n=2,
            axis=1,
            data=xt.iTuple.one(
                xf.Loc.result(INPUT, 0)
            )
        ))
        .add_node(GP, xf.gp.GP_RBF(
            # sigma=1.,
            sites_features=xt.iTuple.one(
                xf.Loc.result(LATENT, 0),
            ),
            sites_data=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            )
        ))
        .add_constraint(xf.nodes.constraints.loss.Constraint_MSE(
            data=xt.iTuple(
                xf.Loc.result(COV, 0),
                xf.Loc.result(GP, 0),
            )
        ))
        # .add_constraint(xf.nodes.constraints.loss.Constraint_MSE(
        #     data=xt.iTuple(
        #         xf.Loc.result(INPUT, 0),
        #         xf.Loc.result(GP, 0, 1),
        #     )
        # ))
        .init(data)
    )

    model = model.optimise(data, iters = 2500)
    results = model.apply(data)

    params = model.params

    latents = params[LATENT][0]

    k_means = KMeans(n_clusters=3, random_state=69).fit(latents)
    
    labels = xt.iTuple(k_means.labels_)
    labels_ordered = {
        label: i for i, label in enumerate(sorted(
            set(labels),
            key=labels.index
        ))
    }
    labels = labels.map(lambda l: labels_ordered[l])
    label_map = {i: l for i, l in labels.enumerate()}

    assert label_map == CLUSTER_MAP, dict(
        labels=label_map,
        clusters=CLUSTER_MAP,
    )

    cov_res = results[GP][0]

    utils.assert_is_close(
        cov,
        cov_res,
        True,
        atol=.05,
        n_max=10,
    )

    return True
