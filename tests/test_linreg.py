
import datetime

import numpy
import pandas

import xtuples as xt
import src.xfactors as xf

from tests import utils

def test_linreg():

    ds = xf.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_i = xf.rand.gaussian((100, 3,))
    betas = xf.rand.gaussian((3, 1,))
    vs_o = numpy.matmul(vs_i, betas)

    data = (
        pandas.DataFrame({
            f: xf.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs_i).T)
        }),
        pandas.DataFrame({
            f: xf.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs_o).T)
        }),
    )

    model, STAGES = xf.Model().init_stages(1)
    INPUT, REGRESS = STAGES

    model = (
        model.add_input(xf.inputs.Input_DataFrame_Wide())
        .add_input(xf.inputs.Input_DataFrame_Wide())
        .add_operator(REGRESS, xf.reg.Lin_Reg(
            n=1,
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            #
        ))
        .add_constraint(xf.constraints.Constraint_MSE(
            sites=xt.iTuple(
                xf.Loc.result(INPUT, 1),
                xf.Loc.result(REGRESS, 0),
            )
        ))
        .init_shapes_params(data)
    )

    betas_pre = model.params[REGRESS][0].T

    model = model.optimise(data)
    results = model.apply(data)
    params = model.params

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
