
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

    N = 100

    ds = xf.utils.dates.starting(datetime.date(2020, 1, 1), N)

    process = numpy.array(xf.utils.rand.gaussian((N,)))
    v_series = xf.utils.dates.dated_series({
        d: v for d, v in zip(ds, process) #
    })
    # .rolling("5D").mean().fillna(0)

    noise = xf.utils.rand.gaussian((N,)) / 3

    factors = (
        pandas.DataFrame({
            0: v_series,
            1: pandas.Series(
                index=v_series.index,
                data=numpy.concatenate([
                    numpy.zeros(1),
                    numpy.cumsum(v_series.values)[:-1],
                ]) + noise,
            )
        }),
    )

    N_COLS = 3
    betas = xf.utils.rand.gaussian((1, N_COLS,))

    vs = numpy.matmul(factors[0].values[:, 1:], betas)
    vs = vs + xf.utils.rand.gaussian(vs.shape) / 3

    data = (
        pandas.DataFrame({
            f: xf.utils.dates.dated_series({
                d: v for d, v in zip(ds, fvs) #
            })
            for f, fvs in enumerate(numpy.array(vs).T)
        }),
    )

    model = xf.Model()
    model, loc_data = model.add_node(
        xf.inputs.dfs.DataFrame_Wide(),
        input=True,
    )

    model, loc_state = model.add_node(
        xf.params.random.Gaussian((N, 2,))
    )
    model, loc_cov = model.add_node(
        xf.params.random.Orthogonal((N, 2, 2,))
    )
    model, loc_transition = model.add_node(
        xf.params.random.Gaussian((2, 2,))
    )
    model, loc_observation = model.add_node(
        xf.params.random.Gaussian((N_COLS, 2,))
    )
    
    model, loc_noise_state_eigvec = model.add_node(
        xf.params.random.Orthogonal((2, 2,))
    )
    model, loc_noise_obs_eigvec = model.add_node(
        xf.params.random.Orthogonal((3, 3,))
    )
    model, loc_noise_state_eigval_raw = model.add_node(
        xf.params.random.Gaussian((2,))
    )
    model, loc_noise_obs_eigval_raw = model.add_node(
        xf.params.random.Gaussian((3,))
    )
    model, loc_noise_state_eigval = model.add_node(
        xf.transforms.scaling.Sq(
            loc_noise_state_eigval_raw.param()
        )
    )
    model, loc_noise_obs_eigval = model.add_node(
        xf.transforms.scaling.Sq(
            loc_noise_obs_eigval_raw.param()
        )
    )
    model, loc_noise_state = model.add_node(
        xf.transforms.linalg.Eigen_Cov(
            loc_noise_state_eigval.result(),
            loc_noise_state_eigvec.param(),
        )
    )
    model, loc_noise_obs = model.add_node(
        xf.transforms.linalg.Eigen_Cov(
            loc_noise_obs_eigval.result(),
            loc_noise_obs_eigvec.param(),
        )
    )

    model, loc_state_pred = model.add_node(
        xf.forecasting.kf.State_Prediction(
            transition=loc_transition.param(),
            state=loc_state.param(),
            # control = loc_control.param(),
            # input=loc_input.param(),
        )
    )
    model, loc_cov_pred = model.add_node(
        xf.forecasting.kf.Cov_Prediction(
            transition=loc_transition.param(),
            cov=loc_cov.param(),
            noise=loc_noise_state.result()
        )
    )
    model, loc_state_inn = model.add_node(
        xf.forecasting.kf.State_Innovation(
            data=loc_data.result(),
            observation=loc_observation.param(),
            state=loc_state_pred.result(),
        )
    )
    model, loc_cov_inn = model.add_node(
        xf.forecasting.kf.Cov_Innovation(
            observation=loc_observation.param(),
            cov=loc_cov_pred.result(),
            noise = loc_noise_obs.result(),
        )
    )
    model, loc_kg = model.add_node(
        xf.forecasting.kf.Kalman_Gain(
            observation=loc_observation.param(),
            cov=loc_cov_pred.result(),
            cov_innovation=loc_cov_inn.result(),
        )
    )
    model, loc_state_new = model.add_node(
        xf.forecasting.kf.State_Updated(
            state=loc_state_pred.result(),
            kalman_gain=loc_kg.result(),
            state_innovation=loc_state_inn.result(),
        ),
        markov=loc_state.param(),
    )
    model, loc_cov_new = model.add_node(
        xf.forecasting.kf.Cov_Updated(
            kalman_gain=loc_kg.result(),
            observation=loc_observation.param(),
            cov=loc_cov_pred.result(),
        ),
        markov=loc_cov.param(),
    )
    model, loc_residual = model.add_node(
        xf.forecasting.kf.Residual(
            data=loc_data.result(),
            observation=loc_observation.param(),
            state=loc_state_new.result(),
        )
    )
    model = (
        model.add_node(
            xf.constraints.loss.MinimiseSquare(
                loc_state_inn.result()
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.Eigenvec_Cov(
                loc_noise_state_eigval.result(),
                loc_noise_state_eigvec.result(),
            ),
            constraint=True,
        )
        .add_node(
            xf.constraints.linalg.Eigenvec_Cov(
                loc_noise_obs_eigval.result(),
                loc_noise_obs_eigvec.result(),
            ),
            constraint=True,
        )
    ).init(data)

    model = model.optimise(
        data, 
        iters = 2500,
        max_error_unchanged=0.5,
        rand_init=100,
        # opt = optax.sgd(.1),
        # opt=optax.noisy_sgd(.1),
        # jit=False,
    ).apply(data)

    transition = numpy.round(
        loc_transition.param().access(model), 2
    )
    observation = numpy.round(
        loc_observation.param().access(model), 2
    )

    noise_obs = numpy.round(
        loc_noise_obs.result().access(model), 2
    )
    noise_state = numpy.round(
        loc_noise_state.result().access(model), 2
    )

    for res in [
        numpy.round(betas),
        transition,
        observation,
        noise_obs,
        noise_state,
    ]:
        print(res)

    assert False

    # target transition model =
    # new_0 = [0 * old_0, 0 * old_1]
    # new_1 = [1 * old_1, 1 * old_0]

    # target observation model = betas

    # loc_noise_state = 
    # diagonal
    # 0: variance = 1
    # 1: varaince = 1 / 3 (?)

    # loc_noise_obs = 
    # diagonal
    # variance = 1 / 3

    return True 