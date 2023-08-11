
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xfactors as xf

from tests import utils

def test_ppca() -> bool:

    N = 3

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)
    vs_norm = xf.utils.rand.gaussian((100, N,))

    betas = xf.utils.rand.gaussian((N, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model, STAGES = xf.Model().init_stages(4)
    INPUT, COV, ENCODE, DECODE, EM = STAGES

    model = (
        model.add_input(xf.nodes.inputs.dfs.Input_DataFrame_Wide())
        .add_node(COV, xf.nodes.cov.vanilla.Cov(
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ), static=True,
        ))
        .add_node(ENCODE, xf.nodes.pca.vanilla.PCA_Encoder(
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            n=N,
            train=False,
            #
        ))
        .add_node(ENCODE, xf.nodes.params.scalar.Scalar(
            v=jax.numpy.ones(1),
        ))
        .add_node(DECODE, xf.nodes.pca.vanilla.PCA_Decoder(
            sites=xt.iTuple(
                xf.Loc.param(ENCODE, 0),
                xf.Loc.result(ENCODE, 0),
            ),
            train=False,
            #
        ))
        .add_node(EM, xf.nodes.pca.vanilla.PPCA_EM(
            site_sigma=xf.Loc.param(ENCODE, 1),
            sites_weights=xt.iTuple.one(
                xf.Loc.param(ENCODE, 0),
            ),
            site_cov=xf.Loc.result(COV, 0),
            train=True,
            # random=0.01,
        ))
        .add_constraint(xf.nodes.constraints.linalg.Constraint_Orthonormal(
            sites=xf.xt.iTuple.one(
                xf.Loc.param(ENCODE, 0),
            ),
            T=True,
        ))
        .add_constraint(xf.nodes.constraints.em.Constraint_EM(
            sites_param=xt.iTuple.one(
                xf.Loc.param(ENCODE, 0)
            ),
            sites_optimal=xt.iTuple.one(
                xf.Loc.result(EM, 0, 0)
            ),
            # cut_tree=True,
        ))
        .add_constraint(xf.nodes.constraints.em.Constraint_EM(
            sites_param=xt.iTuple.one(
                xf.Loc.param(ENCODE, 1)
            ),
            sites_optimal=xt.iTuple.one(
                xf.Loc.result(EM, 0, 1)
            ),
            # cut_tree=True,
        ))
        .init(data)
    )

    model = model.optimise(
        data,
        iters = 2500,
        max_error_unchanged = 500,
        # opt = optax.sgd(.1),
        opt=optax.noisy_sgd(.1),
    )
    results = model.apply(data)
    params = model.params

    eigen_vec = params[ENCODE][0]
    sigma = params[ENCODE][1]

    factors = results[ENCODE][0]

    cov = jax.numpy.cov(factors.T)
    eigen_vals = jax.numpy.diag(cov)

    order = numpy.flip(numpy.argsort(eigen_vals))[:N]
    assert eigen_vals.shape[0] == N, eigen_vals.shape

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]
    
    _, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))
    
    utils.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True
