
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
    
    model, STAGES = xf.Model().init_stages(2)
    INPUT, PARAMS, GMM = STAGES

    model = (
        model.add_input(xf.nodes.inputs.dfs.Input_DataFrame_Wide())
        .add_node(PARAMS, xf.nodes.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        ))
        .add_node(PARAMS, xf.nodes.params.random.Gaussian(
            shape=(N_CLUSTERS, N_COLS, N_COLS,),
        ))
        # .add_node(PARAMS, xf.nodes.params.random.GaussianSoftmax(
        #     shape=(data[0].shape[0], N_CLUSTERS,),
        # ))
        .add_node(PARAMS, xf.nodes.params.random.GaussianSoftmax(
            shape=(N_CLUSTERS,),
        ))
        .add_node(GMM, xf.nodes.clustering.gmm.BGMM_EM(
            k=N_CLUSTERS,
            data=xf.Loc.result(INPUT, 0),
            mu=xf.Loc.param(PARAMS, 0),
            cov=xf.Loc.param(PARAMS, 1),
        ), random = True)
        .add_constraint(xf.nodes.constraints.loss.Constraint_Maximise(
            data=xf.Loc.result(GMM, 0, 1),
        ))
        .add_constraint(xf.nodes.constraints.loss.Constraint_Maximise(
            data=xf.Loc.result(GMM, 0, 2),
        ))
        .add_constraint(xf.nodes.constraints.linalg.Constraint_VOrthogonal(
            data=xf.Loc.param(PARAMS, 1),
        ))
        .add_constraint(xf.nodes.constraints.linalg.Constraint_L1_MM_Diag(
            raw=xf.Loc.param(PARAMS, 1),
        ))
        .init(data)
    )

    # from jax.config import config 
    # config.update("jax_debug_nans", True) 

    model = model.optimise(
        data,
        iters = 1000,
        opt=optax.noisy_sgd(.1),
        max_error_unchanged = 0.3,
        rand_init=1000,
        # jit = False,
    )
    results = model.apply(data)

    params = model.params[PARAMS]

    mu_ = params[0]
    cov_ = params[1]
    # probs = params[2]
    probs = results[GMM][0][0]
    
    cov_ = numpy.round(numpy.matmul(
        numpy.transpose(cov_, (0, 2, 1)),
        cov_,
    ), 3)

    labels = probs.argmax(axis=1)
    # n_data

    print(cov_)
    print(labels)
    print(mu_)
    print(mu)
    
    # print(results[EM][0][3])
    # print(results[EM][0][0])
    
    labels, order = (
        xt.iTuple([int(l) for l in labels])
        .pipe(xf.nodes.clustering.kmeans.reindex_labels)
    )
    mu_ = [mu_[i] for i in order]

    k_means = KMeans(n_clusters=3, random_state=69).fit(vs)
    sk_labels, sk_order = xt.iTuple(k_means.labels_).pipe(
        xf.nodes.clustering.kmeans.reindex_labels
    )

    mu_ = numpy.round(mu_, 3)
    mu = numpy.round(mu, 3)
    
    assert labels == sk_labels, {
        i: (l, sk_l,) for i, (l, sk_l)
        in enumerate(zip(labels, sk_labels))
        if l != sk_l
    }

    utils.assert_is_close(
        mu_,
        mu,
        True,
        atol=0.2,
    )

    return True