
import datetime

import numpy
import pandas
import jax

import xtuples as xt
import src.xfactors as xf

from tests import utils

def test_ppca():

    N = 3

    ds = xf.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xf.rand.gaussian((100, N,))
    betas = xf.rand.gaussian((N, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xf.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model, STAGES = xf.Model().init_stages(3)
    INPUT, COV, ENCODE, DECODE = STAGES

    model = (
        model.add_input(xf.inputs.Input_DataFrame_Wide())
        .add_operator(COV, xf.stats.Cov(
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ), static=True,
        ))
        .add_operator(ENCODE, xf.pca.PCA_Encoder(
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            n=N + 1,
            #
        ))
        .add_operator(DECODE, xf.pca.PCA_Decoder(
            sites=xt.iTuple(
                xf.Loc.param(ENCODE, 0),
                xf.Loc.result(ENCODE, 0),
            )
            #
        ))
        .add_constraint(xf.constraints.Constraint_MSE(
            sites=xt.iTuple(
                xf.Loc.result(INPUT, 0),
                xf.Loc.result(DECODE, 0),
            )
        ))
        .add_constraint(xf.constraints.Constraint_EigenVLike(
            sites=xf.xt.iTuple(
                xf.Loc.param(ENCODE, 0),
                xf.Loc.result(ENCODE, 0),
            ),
            n_check=N + 1,
        ))
        .init_shapes_params(data)
    )

    model = model.optimise(data)
    results = model.apply(data)
    params = model.params

    eigen_vec = params[ENCODE][0]
    factors = results[ENCODE][0]

    cov = jax.numpy.cov(factors.T)
    eigen_vals = jax.numpy.diag(cov)

    order = numpy.flip(numpy.argsort(eigen_vals))[:N]
    assert eigen_vals.shape[0] == N + 1, eigen_vals.shape

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))
    # assert False, (eigvals, eigen_vals,)

    print(eigen_vec)
    print(eigvecs)

    # for now we just check pc1 matches
    utils.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True
