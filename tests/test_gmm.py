
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

    N_COLS = 5
    N_CLUSTERS = 3
    N_VARIABLES = 30

    mu = numpy.stack([
        numpy.ones(N_COLS) * -1,
        numpy.zeros(N_COLS),
        numpy.ones(N_COLS) * 1,
    ]) + (xf.rand.gaussian((N_CLUSTERS, N_COLS,)) / 2)

    vs = numpy.concatenate([
        mu[cluster] + (xf.rand.gaussian((N_VARIABLES, N_COLS)) / 2)
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
    INPUT, PARAMS, EM = STAGES

    model = (
        model.add_input(xf.inputs.Input_DataFrame_Wide())
        .add_operator(PARAMS, xf.params.Gaussian(
            shape=(N_CLUSTERS, N_COLS,),
        ))
        .add_operator(PARAMS, xf.params.Gaussian(
            shape=(N_CLUSTERS, N_COLS, N_COLS,),
        ))
        # .add_operator(PARAMS, xf.params.RandomCovariance(
        #     n=N_CLUSTERS,
        #     d=N_COLS,
        # ))
        .add_operator(PARAMS, xf.params.GaussianSoftmax(
            shape=(data[0].shape[0], N_CLUSTERS,),
        ))
        .add_operator(PARAMS, xf.params.GaussianSoftmax(
            shape=(N_CLUSTERS,),
        ))
        .add_operator(EM, xf.mixtures.BGMM_EM(
            k=N_CLUSTERS,
            sites_data=xt.iTuple.one(
                xf.Loc.result(INPUT, 0)
            ),
            sites_mu=xt.iTuple.one(
                xf.Loc.param(PARAMS, 0)
            ),
            sites_cov=xt.iTuple.one(
                xf.Loc.param(PARAMS, 1)
            ),
            sites_probs=xt.iTuple.one(
                xf.Loc.param(PARAMS, 3)
            ),
        ))
        # .add_constraint(xf.constraints.Constraint_MinimiseSquare(
        #     sites=xt.iTuple.one(
        #         xf.Loc.result(EM, 0, 3)
        #     ),
        # ))
        # .add_constraint(xf.constraints.Constraint_Orthogonal(
        #     sites=xt.iTuple.one(
        #         xf.Loc.result(EM, 0, 0)
        #     ),
        # ))
        .add_constraint(xf.constraints.Constraint_Maximise(
            sites=xt.iTuple.one(
                xf.Loc.result(EM, 0, 3)
            ),
        ))
        .add_constraint(xf.constraints.Constraint_Maximise(
            sites=xt.iTuple.one(
                xf.Loc.result(EM, 0, 4)
            ),
        ))
        # .add_constraint(xf.constraints.Constraint_MinimiseSquare(
        #     sites=xt.iTuple.one(
        #         xf.Loc.result(EM, 0)
        #     ),
        # ))
        # .add_constraint(xf.constraints.Constraint_Orthogonal(
        #     sites=xf.xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 0)
        #     ),
        # ))
        .add_constraint(xf.constraints.Constraint_VOrthogonal(
            sites=xf.xt.iTuple.one(
                xf.Loc.param(PARAMS, 1)
            ),
        ))
        # .add_constraint(xf.constraints.Constraint_MinimiseMMSpread(
        #     sites=xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 1)
        #     ),
        # ))
        .add_constraint(xf.constraints.Constraint_L1_MM_Diag(
            sites=xt.iTuple.one(
                xf.Loc.param(PARAMS, 1)
            ),
        ))
        # .add_constraint(xf.constraints.Constraint_Orthogonal(
        #     sites=xf.xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 0)
        #     ),
        # ))
        # .add_constraint(xf.constraints.Constraint_EM(
        #     sites_param=xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 0)
        #     ),
        #     sites_optimal=xt.iTuple.one(
        #         xf.Loc.result(EM, 0, 0)
        #         #mu
        #     ),
        #     cut_tree=True,
        # ))
        # .add_constraint(xf.constraints.Constraint_EM_MatMul(
        #     sites_param=xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 1)
        #     ),
        #     sites_optimal=xt.iTuple.one(
        #         xf.Loc.result(EM, 0, 1)
        #         #a
        #     ),
        #     cut_tree=True,
        # ))
        # .add_constraint(xf.constraints.Constraint_EM(
        #     sites_param=xt.iTuple.one(
        #         xf.Loc.param(PARAMS, 2)
        #     ),
        #     sites_optimal=xt.iTuple.one(
        #         xf.Loc.result(EM, 0, 3)
        #         # b
        #     ),
        #     cut_tree=True,
        # ))
        .init_shapes_params(data)
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

    clusters = params[0]
    cov_ = params[1]
    # probs = params[2]
    probs = results[EM][0][2]
    
    cov_ = numpy.round(numpy.matmul(
        numpy.transpose(cov_, (0, 2, 1)),
        cov_,
    ), 3)

    labels = probs.argmax(axis=1)
    # n_data

    print(cov_)
    print(labels)
    print(clusters)
    print(mu)
    
    # print(results[EM][0][3])
    # print(results[EM][0][0])
    
    labels, order = (
        xt.iTuple([int(l) for l in labels])
        .pipe(xf.kmeans.reindex_labels)
    )
    clusters = [clusters[i] for i in order]

    k_means = KMeans(n_clusters=3, random_state=69).fit(vs)
    sk_labels, sk_order = xt.iTuple(k_means.labels_).pipe(
        xf.kmeans.reindex_labels
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
