
import datetime

import numpy
import pandas

import xtuples as xt

from src.xfactors import rand
from src.xfactors import dates
from src.xfactors import xfactors as xf

from tests import utils

def test_linreg():

    ds = dates.starting(datetime.date(2020, 1, 1), 100)

    vs_i = rand.normal((100, 3,))
    betas = rand.normal((3, 1,))
    vs_o = numpy.matmul(vs_i, betas)

    data = (
        pandas.DataFrame({
            f: dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs_i).T)
        }),
        pandas.DataFrame({
            f: dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs_o).T)
        }),
    )

    STAGES = xf.init_stages(1)
    INPUT, REGRESS = STAGES

    model, params, results, objective, apply = (
        xf.Model()
        .add_input(xf.Input_DataFrame_Wide())
        .add_input(xf.Input_DataFrame_Wide())
        .add_stage()
        .add_operator(REGRESS, xf.Lin_Reg(
            n=1,
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            #
        ))
        .add_constraint(xf.Constraint_MSE(
            sites=xt.iTuple(
                xf.Loc.result(INPUT, 1),
                xf.Loc.result(REGRESS, 0),
            )
        ))
        .build(data)
    )

    betas_pre = params[REGRESS][0].T

    model, params = model.optimise(
        params, results, objective, verbose=False
    )
    betas_post = params[REGRESS][0].T

    results = dict(
        betas=betas.squeeze(),
        pre=betas_pre.squeeze(),
        post=betas_post.squeeze(),
    )

    utils.assert_is_close(
        results["betas"],
        results["pre"],
        False,
        results,
    )
    utils.assert_is_close(
        results["betas"],
        results["post"],
        True,
        results,
    )

    return True
