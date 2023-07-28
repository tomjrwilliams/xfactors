
import datetime

import numpy
import pandas

import xtuples as xt

from src.xfactors import rand
from src.xfactors import dates
from src.xfactors import xfactors as xf

from tests import utils

def test_pca():

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

    STAGES = xf.init_stages(1)
    INPUT, PCA = STAGES

    model, params, results, objective, apply = (
        xf.Model()
        .add_input(xf.Input_DataFrame_Wide())
        .add_stage()
        .add_operator(PCA, xf.PCA(
            n=3,
            sites=xt.iTuple.one(
                xf.Loc.result(INPUT, 0),
            ),
            #
        ))
        .build(data)
    )

    model, params = model.optimise(
        params, results, objective, verbose=False
    )
    results = apply(params, data)
    
    eigen_val = results[PCA][0][0]
    eigen_vec = results[PCA][0][1]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))

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
