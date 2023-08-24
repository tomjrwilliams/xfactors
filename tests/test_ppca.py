
import datetime

import numpy
import pandas
import jax
import optax

import xtuples as xt
import xfactors as xf

from tests import utils

def test_ppca() -> bool:
    xf.utils.rand.reset_keys()

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
    NOISE = 0

    model, loc_data = xf.Model().add_node(
        xf.nodes.inputs.dfs.Input_DataFrame_Wide(),
        input=True,
    )
    model, loc_cov = model.add_node(xf.nodes.cov.vanilla.Cov(
        data=loc_data.result()
    ), static=True)

    model = (
        model.add_node(PARAMS, xf.nodes.params.random.Gaussian(
            (N + NOISE,),
        ))
        .add_node(SCALING, xf.nodes.scaling.scalar.Scale_Sq(
            data=xf.Loc.param(PARAMS, 0),
        ))
        .add_node(ENCODE, xf.nodes.pca.vanilla.PCA_Encoder(
            data=xf.Loc.result(INPUT, 0),
            n=N + NOISE,
            #
        ))
        .add_node(DECODE, xf.nodes.pca.vanilla.PCA_Decoder(
            weights=xf.Loc.param(ENCODE, 0),
            factors=xf.Loc.result(ENCODE, 0),
            #
        ))
        # .add_constraint(xf.nodes.constraints.linalg.Constraint_Orthonormal(
        #     data=xf.Loc.param(ENCODE, 0),
        #     T=True,
        # ), not_if=dict(init=True))
        .add_constraint(xf.nodes.constraints.linalg.Constraint_Eigenvec(
            cov=xf.Loc.result(COV, 0),
            weights=xf.Loc.param(ENCODE, 0),
            eigvals=xf.Loc.result(SCALING, 0),
        ))
        .init(data)
    )

    model = model.optimise(
        data, 
        iters = 2500,
        # max_error_unchanged=1000,
        rand_init=100,
        # opt = optax.sgd(.1),
        # opt=optax.noisy_sgd(.1),
        # jit=False,
    )
    results = model.apply(data)
    params = model.params

    eigen_vec = params[ENCODE][0]
    sigma = results[SCALING][0]

    factors = results[ENCODE][0]

    # cov = jax.numpy.cov(factors.T)
    # eigen_vals = jax.numpy.diag(cov)

    print(numpy.round(
        numpy.matmul(eigen_vec.T, eigen_vec),
        2
    ))

    eigen_vals = sigma

    order = numpy.flip(numpy.argsort(eigen_vals))
    # assert eigen_vals.shape[0] == N, eigen_vals.shape

    eigen_vals = eigen_vals[order]
    eigen_vec = eigen_vec[..., order]

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(
        numpy.transpose(data[0].values)
    ))
    _order = numpy.flip(numpy.argsort(eigvals))
    eigvecs = eigvecs[..., _order]
    eigvals = eigvals[_order]

    print(numpy.round(eigen_vec, 4))
    print(numpy.round(eigvecs, 4))

    print(numpy.round(eigen_vals, 3))
    print(numpy.round(eigvals, 3))

    utils.assert_is_close(
        eigen_vec.real[..., :1],
        eigvecs.real[..., :1],
        True,
        atol=.1,
    )
    return True