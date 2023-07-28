
import datetime

import numpy
import pandas
import jax

import xtuples as xt

from src.xfactors import rand
from src.xfactors import dates
from src.xfactors import xfactors as xf

from tests import utils

def test_ppca():

    N = 3

    ds = dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = rand.normal((100, N,))
    betas = rand.normal((N, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    STAGES = xf.init_stages(2)
    INPUT, ENCODE, DECODE = STAGES

    model, params, results, objective, apply = (
        xf.Model()
        .add_input(xf.Input_DataFrame_Wide())
        .add_stage()
        .add_operator(ENCODE, xf.PCA_Encoder(
            n=N,
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            #
        ))
        .add_stage()
        .add_operator(DECODE, xf.PCA_Decoder(
            sites=xt.iTuple(
                xf.Loc.param(ENCODE, 0),
                xf.Loc.result(ENCODE, 0),
            )
            #
        ))
        .add_constraint(xf.Constraint_MSE(
            sites=xt.iTuple(
                xf.Loc.result(INPUT, 0),
                xf.Loc.result(DECODE, 0),
            )
        ))
        .add_constraint(xf.Constraint_EigenVLike(
            sites=xf.xt.iTuple(
                xf.Loc.param(ENCODE, 0),
                xf.Loc.result(ENCODE, 0),
            ),
            n_check=N,
        ))
        .build(data)
    )

    model, params = model.optimise(params, results, objective)
    results = apply(params, data)

    eigen_vec = params[ENCODE][0]
    factors = results[ENCODE][0]

    cov = jax.numpy.cov(factors.T)
    eigen_vals = jax.numpy.diag(cov)

    assert eigen_vals.shape[0] == N, eigen_vals.shape

    order = numpy.flip(numpy.argsort(eigen_vals))

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))

    # for now we just check pc1 matches
    utils.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True
