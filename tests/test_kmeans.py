
import itertools
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf

from tests import utils

import optax

# import jax.config
# jax.config.update("jax_debug_nans", True)

from sklearn.cluster import KMeans

def test_kmeans() -> bool:
    xf.utils.rand.reset_keys()

    N_COLS = 5
    N_CLUSTERS = 3
    N_VARIABLES = 30

    mu = numpy.stack([
        numpy.ones(N_COLS) * -1,
        numpy.zeros(N_COLS),
        numpy.ones(N_COLS) * 1,
    ]) + (xf.utils.rand.gaussian((N_CLUSTERS, N_COLS,)) / 2)

    vs = numpy.concatenate([
        mu[cluster] + (xf.utils.rand.gaussian((N_VARIABLES, N_COLS)) / 2)
        for cluster in range(N_CLUSTERS)
    ], axis = 0)

    data = (
        pandas.DataFrame({
            f: pandas.Series(
                index=list(range(len(fvs))),
                data=fvs,
            )
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    
    model, STAGES = xf.Model().init_stages(3)
    INPUT, PARAMS, LABEL, EM = STAGES

    model = (
        model.add_input(xf.nodes.inputs.dfs.Input_DataFrame_Wide())
        .add_node(PARAMS, xf.nodes.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        ))
        .add_node(PARAMS, xf.nodes.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        ))
        .add_node(LABEL, xf.nodes.clustering.kmeans.KMeans_Labels(
            k=3,
            mu=xf.Loc.param(PARAMS, 0),
            var=xf.Loc.param(PARAMS, 1),
            data=xf.Loc.result(INPUT, 0),
            # mu
            # cov
        ))
        .add_node(EM, xf.nodes.clustering.kmeans.KMeans_EM_Naive(
            k=3,
            data=xf.Loc.result(INPUT, 0),
            labels=xf.Loc.result(LABEL, 0),
        ))
        .add_constraint(xf.nodes.constraints.em.Constraint_EM(
            param=xf.Loc.param(PARAMS, 0),
            optimal=xf.Loc.result(EM, 0, 0),
            cut_tree=True,
        ))
        .add_constraint(xf.nodes.constraints.em.Constraint_EM(
            param=xf.Loc.param(PARAMS, 1),
            optimal=xf.Loc.result(EM, 0, 1),
            cut_tree=True,
        ))
        .init(data)
    )

    model = model.optimise(
        data,
        iters = 1000,
        opt=optax.noisy_sgd(.1),
        rand_init=100,
        # jit = False,
    )
    results = model.apply(data)

    params = model.params

    clusters = params[PARAMS][0]
    
    labels, order = (
        xt.iTuple([int(l) for l in results[LABEL][0]])
        .pipe(xf.nodes.clustering.kmeans.reindex_labels)
    )
    clusters = [clusters[i] for i in order]

    k_means = KMeans(n_clusters=3, random_state=69).fit(vs)
    sk_labels, sk_order = xt.iTuple(k_means.labels_).pipe(
        xf.nodes.clustering.kmeans.reindex_labels
    )

    clusters = numpy.round(clusters, 3)
    mu = numpy.round(mu, 3)
    
    assert labels == sk_labels, {
        i: (l, sk_l,) for i, (l, sk_l)
        in enumerate(zip(labels, sk_labels))
        if l != sk_l
    }

    utils.assert_is_close(
        clusters,
        mu,
        True,
        atol=0.2,
    )

    return True