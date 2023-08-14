
import datetime

import numpy
import pandas

import xtuples as xt
import xfactors as xf

from tests import utils

def test_pca() -> bool:
    xf.utils.rand.reset_keys()

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), 100)

    vs_norm = xf.utils.rand.gaussian((100, 3,))
    betas = xf.utils.rand.gaussian((3, 5,))
    vs = numpy.matmul(vs_norm, betas)

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({d: v for d, v in zip(ds, fvs)})
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )
    
    model, STAGES = xf.Model().init_stages(1)
    INPUT, PCA = STAGES

    model = (
        model.add_input(xf.nodes.inputs.dfs.Input_DataFrame_Wide())
        .add_node(PCA, xf.nodes.pca.vanilla.PCA(
            n=3,
            data=xf.Loc.result(INPUT, 0),
            #
        ))
        .init(data)
    )

    model = model.optimise(data)
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