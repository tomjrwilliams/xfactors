
import datetime

import numpy
import pandas

import xtuples as xt
import src.xfactors as xf

from tests import utils

def test_pca():

    ds = xf.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xf.rand.gaussian((100, 3,))
    betas = xf.rand.gaussian((3, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xf.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    
    model, STAGES = xf.Model().init_stages(1)
    INPUT, PCA = STAGES

    model, objective = (
        model.add_input(xf.inputs.Input_DataFrame_Wide())
        .add_stage()
        .add_operator(PCA, xf.pca.PCA(
            n=3,
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            #
        ))
        .build(data)
    )

    model = model.optimise(objective)
    results = model.apply(data)
    params = model.params
    
    eigen_val = results[PCA][0][0]
    eigen_vec = results[PCA][0][1]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))

    # multiply by root(eigenval) -> beta?

    utils.assert_is_close(
        eigen_val[:3],
        eigvals.real[:3],
        True,
    )
    utils.assert_is_close(
        eigen_vec.real[..., :3],
        eigvecs.real[..., :3],
        True,
    )

    return True
