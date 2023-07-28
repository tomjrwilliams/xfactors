
import datetime

import numpy
import pandas
import jax

import xtuples as xt

from src.xfactors import rand
from src.xfactors import dates
from src.xfactors import xfactors as xf

def test_ppca():

    ds = dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = rand.normal((100, 3,))
    betas = rand.normal((3, 5,))
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
            n=3,
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
            )
        ))
        .build(data)
    )

    model, params = model.optimise(params, results, objective)

    # assert is the same results as if we did an explicit pca

    return dict(
        betas=betas,
        factors=params[ENCODE][0].T,
    )
