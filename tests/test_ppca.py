
import datetime

import numpy
import pandas
import jax

from src.xfactors import rand
from src.xfactors import dates
from src.xfactors import xfactors

def test_ppca():

    ds = dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = rand.normal((100, 3,))
    betas = rand.normal((3, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(vs.T)
        }),
    )

    model, params, results, objective, apply = (
        xfactors.Model()
        .add_input(xfactors.Input_DataFrame_Wide(
            
        ))
        .add_factor(xfactors.Factor_PCA(
            n=3,
            sites=xfactors.sites(
                xfactors.loc_result(xfactors.Stage.INPUT, 0),
            ),
        ))
        .add_output(xfactors.Output_PCA(
            sites=xfactors.sites(
                xfactors.loc_param(xfactors.Stage.FACTOR, 0),
                xfactors.loc_result(xfactors.Stage.FACTOR, 0),
            )
        ))
        .add_constraint(xfactors.Constraint_MSE(
            sites=xfactors.sites(
                xfactors.loc_result(xfactors.Stage.INPUT, 0),
                xfactors.loc_result(xfactors.Stage.OUTPUT, 0),
            )
        ))
        .add_constraint(xfactors.Constraint_EigenVLike(
            sites=xfactors.sites(
                xfactors.loc_param(xfactors.Stage.FACTOR, 0),
                xfactors.loc_result(xfactors.Stage.FACTOR, 0),
            )
        ))
        .build(data)
    )

    model, params = model.optimise(params, results, objective)

    return dict(
        betas=betas,
        factors=params.factors[0].T,
    )
