import enum

import collections
# import collections.abc

import functools
import itertools

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import jaxopt
import optax

import xtuples

from . import rand
from . import dates

# ---------------------------------------------------------------

from jax.config import config 
# config.update("jax_debug_nans", True) 

# ---------------------------------------------------------------

class Orientation(enum.Enum):
    WIDE: int = 0
    TALL: int = 1

class Domain(enum.Enum):
    PARAM: int = 0
    RESULT: int = 1

class Stage(enum.Enum):
    INPUT: int = 0
    FACTOR: int = 1
    LATENT: int = 2
    OUTPUT: int = 3

# ---------------------------------------------------------------

def loc(domain, stage, i):
    return xtuples.iTuple(domain.value, stage.value, i)

def loc_param(stage, i):
    return loc(Domain.PARAM, stage, i)

def loc_result(stage, i):
    return loc(Domain.RESULT, stage, i)

def sites(*s):
    return xtuples.iTuple(s)

def get_site(s, acc):
    return s.fold(lambda acc, i: acc[i], initial=acc)

def f_get_site(acc):
    def f(s):
        return get_site(s, acc)
    return f

# ---------------------------------------------------------------

def check_input(model, obj):
    return True

def add_input(model, obj):
    assert check_input(model, obj)
    return model._replace(
        inputs = model.inputs.append(obj._replace(
            loc=len(model.inputs)
        ))
        #
    )

def input_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_input,
            add = add_input,
        )
        for k, f in kfs.items():
            methods[k] = f
        for k, f in methods.items():
            setattr(cls, k, f)
        return cls
    return decorator

# ---------------------------------------------------------------

@input_bindings()
@xtuples.nTuple.decorate
class Input_DataFrame(typing.NamedTuple):

    orientation: Orientation = Orientation.WIDE

    loc: int = None
    shape: xtuples.iTuple = None

    def init_shape(self, model, data):
        return data[self.loc].values.shape

    def init_params(self, model, params, results):
        return ()
    
    def apply(self, params_results):
        df = params_results[1][self.loc]
        return df.values

# ---------------------------------------------------------------

def check_factor(model, obj):
    return True

def add_factor(model, obj):
    assert check_factor(model, obj)
    return model._replace(
        factors = model.factors.append(obj._replace(
            loc=len(model.factors)
        ))
        #
    )

def factor_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_factor,
            add = add_factor,
        )
        for k, f in kfs.items():
            methods[k] = f
        for k, f in methods.items():
            setattr(cls, k, f)
        return cls
    return decorator

# ---------------------------------------------------------------

@factor_bindings()
@xtuples.nTuple.decorate
class Factor_PCA(typing.NamedTuple):
    
    n: int
    sites: xtuples.iTuple # [of ituples]

    loc: int = None
    shape: xtuples.iTuple = None

    def init_shape(self, model, data):
        return ()

    def init_params(self, model, params, results):
        objs = self.sites.map(
            lambda loc: loc.tail(2)
        ).map(f_get_site(model))
        shape = (
            objs.map(lambda o: o.shape[1]).pipe(sum),
            self.n,
        )
        return rand.normal(shape)

    def apply(self, params_results):
        weights = params_results[0].factors[self.loc]
        return jax.numpy.matmul(
            jax.numpy.concatenate(
                self.sites.map(f_get_site(params_results)),
                axis=1,
            ),
            weights,
        )


# ---------------------------------------------------------------

def check_latent(model, obj):
    assert obj.axis in [None, 0, 1]
    return True

def add_latent(model, obj):
    assert check_latent(model, obj)
    return model._replace(
        latents = model.latents.append(obj._replace(
            loc=len(model.latents)
        ))
        #
    )

def latent_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_latent,
            add = add_latent,
        )
        for k, f in kfs.items():
            methods[k] = f
        for k, f in methods.items():
            setattr(cls, k, f)
        return cls
    return decorator

# ---------------------------------------------------------------

@latent_bindings()
@xtuples.nTuple.decorate
class Latent(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    axis: int
    # TODO init: collections.abc.Iterable = None

    loc: int = None
    shape: xtuples.iTuple = None

    def init_shape(self, model, data):
        return ()

    def init_params(model, obj, params):
        axis = obj.axis
        shape_latent = (
            (1,) if axis is None
            else (model.shape[axis], 1,) if axis == 0
            else (1, model.shape[axis],) if axis == 1
            else None
        )
        assert shape_latent is not None, obj
        latent = rand.normal(shape_latent)
        return latent

# ---------------------------------------------------------------

def check_output(model, obj):
    return True

def add_output(model, obj):
    assert check_output(model, obj)
    return model._replace(
        outputs = model.outputs.append(obj._replace(
            loc=len(model.outputs)
        ))
        #
    )

def output_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_output,
            add = add_output,
        )
        for k, f in kfs.items():
            methods[k] = f
        for k, f in methods.items():
            setattr(cls, k, f)
        return cls
    return decorator

# ---------------------------------------------------------------

@output_bindings()
@xtuples.nTuple.decorate
class Output_PCA(typing.NamedTuple):
    
    sites: xtuples.iTuple

    loc: int = None
    shape: xtuples.iTuple = None

    def init_shape(self, model, data):
        return ()

    def init_params(self, model, params, results):
        return ()

    def apply(self, params_results):
        assert len(self.sites) == 2
        l_site, r_site = self.sites
        return jax.numpy.matmul(
            get_site(r_site, params_results), 
            get_site(l_site, params_results).T
        )

# ---------------------------------------------------------------

def check_constraint(model, obj):
    return True

def add_constraint(model, obj):
    assert check_constraint(model, obj)
    return model._replace(
        constraints = model.constraints.append(obj._replace(
            loc=len(model.constraints)
        ))
        #
    )

def init_shape_constraint(obj, model, data):
    return

def init_params_constraint(obj, model, data, results):
    return

def constraint_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_constraint,
            add = add_constraint,
            init_shape = init_shape_constraint,
            init_params = init_params_constraint,
        )
        for k, f in kfs.items():
            methods[k] = f
        for k, f in methods.items():
            setattr(cls, k, f)
        return cls
    return decorator

# ---------------------------------------------------------------

def loss_mse(l, r):
    return jax.numpy.square(jax.numpy.subtract(l, r)).mean()

def loss_mse_zero(X1):
    return jax.numpy.square(X1).mean()

@functools.lru_cache(maxsize=4)
def loss_mean_zero(axis):
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    return f

# ascending just reverse order of xl and xr
def loss_descending(x):
    order = jax.numpy.flip(jax.numpy.argsort(x))
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()

def loss_cov_diag(cov, diag):
    diag = jax.numpy.multiply(
        jax.numpy.eye(cov.shape[0]), diag
    )
    return loss_mse(cov, diag)

def loss_orthogonal(X, scale = 1.):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T) / scale
    # return jax.numpy.square(jax.numpy.subtract(XXt, eye)).mean()
    return loss_mse(XXt, eye)

# ---------------------------------------------------------------

@constraint_bindings()
@xtuples.nTuple.decorate
class Constraint_MSE(typing.NamedTuple):
    
    sites: xtuples.iTuple

    loc: int = None
    shape: xtuples.iTuple = None

    def apply(self, params_results):
        assert self.sites.len() == 2
        l_site, r_site = self.sites
        l = get_site(l_site, params_results)
        r = get_site(r_site, params_results)
        return loss_mse(l, r)

@constraint_bindings()
@xtuples.nTuple.decorate
class Constraint_Orthogonal(typing.NamedTuple):
    
    site: xtuples.iTuple

    loc: int = None
    shape: xtuples.iTuple = None

    def apply(self, params_results):
        X = get_site(self.site, params_results)
        return loss_orthogonal(X)

@constraint_bindings()
@xtuples.nTuple.decorate
class Constraint_EigenVLike(typing.NamedTuple):
    
    sites: xtuples.iTuple

    loc: int = None
    shape: xtuples.iTuple = None

    eigval_max: bool = True

    def apply(self, params_results):
        assert len(self.sites) == 2
        w_site, f_site = self.sites
        w = get_site(w_site, params_results)
        f = get_site(f_site, params_results)
        cov = jax.numpy.cov(f.T)
        eigvals = jax.numpy.diag(cov)
        res = (
            + loss_descending(eigvals)
            + loss_orthogonal(w.T)
            + loss_mean_zero(0)(f)
            + loss_cov_diag(cov, eigvals)
        )
        if self.eigval_max:
            return res + (
                - jax.numpy.sum(jax.numpy.log(1 + eigvals))
                # ridge penalty to counteract eigval max
            )
        return res

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Results(typing.NamedTuple):

    inputs: xtuples.iTuple = xtuples.iTuple()
    factors: xtuples.iTuple = xtuples.iTuple()
    latents: xtuples.iTuple = xtuples.iTuple()
    outputs: xtuples.iTuple = xtuples.iTuple()

    def to_tuples(self):
        return self._replace(
            inputs=self.inputs.pipe(tuple),
            factors=self.factors.pipe(tuple),
            latents=self.latents.pipe(tuple),
            outputs=self.outputs.pipe(tuple),
        )

@xtuples.nTuple.decorate
class Params(typing.NamedTuple):

    inputs: xtuples.iTuple = xtuples.iTuple()
    factors: xtuples.iTuple = xtuples.iTuple()
    latents: xtuples.iTuple = xtuples.iTuple()
    outputs: xtuples.iTuple = xtuples.iTuple()
    constraints: xtuples.iTuple = xtuples.iTuple()

    def to_tuples(self):
        return self._replace(
            inputs=self.inputs.pipe(tuple),
            factors=self.factors.pipe(tuple),
            latents=self.latents.pipe(tuple),
            outputs=self.outputs.pipe(tuple),
            constraints=self.constraints.pipe(tuple),
        )

# ---------------------------------------------------------------

def init_shape(model, data):
    model = model._replace(inputs = model.inputs.map(
        lambda o: o._replace(shape=o.init_shape(model, data))
    ))
    model = model._replace(factors = model.factors.map(
        lambda o: o._replace(shape=o.init_shape(model, data))
    ))
    model = model._replace(latents = model.latents.map(
        lambda o: o._replace(shape=o.init_shape(model, data))
    ))
    model = model._replace(outputs = model.outputs.map(
        lambda o: o._replace(shape=o.init_shape(model, data))
    ))
    model = model._replace(constraints = model.constraints.map(
        lambda o: o._replace(shape=o.init_shape(model, data))
    ))
    return model

def init_params(model, data):
    params = Params()
    params = params._replace(inputs = model.inputs.map(
        lambda o: o.init_params(model, params, data)
    ))
    params = params._replace(factors = model.factors.map(
        lambda o: o.init_params(model, params, data)
    ))
    params = params._replace(latents = model.latents.map(
        lambda o: o.init_params(model, params, data)
    ))
    params = params._replace(outputs = model.outputs.map(
        lambda o: o.init_params(model, params, data)
    ))
    params = params._replace(constraints = model.constraints.map(
        lambda o: o.init_params(model, params, data)
    ))
    return params

# ---------------------------------------------------------------

def init_objective(model, init_params, data):
    init_results = Results(inputs = model.inputs.map(
        lambda f: f.apply((init_params, data,))
    ))
    def f(params, results):
        results = results._replace(factors = model.factors.map(
            lambda f: f.apply((params, results,))
        ))
        results = results._replace(latents = model.latents.map(
            lambda f: f.apply((params, results,))
        ))
        results = results._replace(outputs = model.outputs.map(
            lambda f: f.apply((params, results,))
        ))
        return jax.numpy.stack(
            model.constraints.map(
                lambda c: c.apply((params, results,))
            ).pipe(list)
        ).sum()
    return init_results, f

def init_apply(model):
    def f(params, data, sites = None):
        results = Results()
        results = results._replace(inputs = model.inputs.map(
            lambda f: f.apply((init_params, data,))
        ))
        results = results._replace(inputs = model.inputs.map(
            lambda f: f.apply((params, results,))
        ))
        results = results._replace(factors = model.factors.map(
            lambda f: f.apply((params, results,))
        ))
        results = results._replace(latents = model.latents.map(
            lambda f: f.apply((params, results,))
        ))
        results = results._replace(outputs = model.outputs.map(
            lambda f: f.apply((params, results,))
        ))
        if results is None:
            return results
        return {
            k: get_site(s, results)
            for k, s in sites.items()
        }
    return f

# ---------------------------------------------------------------

def build_model(model, data):
    model = model.init_shape(data)
    params = model.init_params(data)
    results, objective = model.init_objective(params, data)
    apply = model.init_apply()
    return model, params, results, objective, apply

def optimise_model(
    model, 
    params, 
    results, 
    objective, 
    lr = 0.01,
    iters=1000,
):
    
    opt = optax.adam(lr)
    solver = jaxopt.OptaxSolver(
        opt=opt, fun=objective, maxiter=iters
    )

    params = params.to_tuples()
    results = results.to_tuples()

    state = solver.init_state(params, results)

    for i in range(iters):
        params, state = solver.update(
            params,
            state,
            results,
        )
        if i % int(iters / 10) == 0 or i == iters - 1:
            print(i, state.error)

    return model, params

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Model(typing.NamedTuple):

    inputs: xtuples.iTuple = xtuples.iTuple()
    factors: xtuples.iTuple = xtuples.iTuple()
    latents: xtuples.iTuple = xtuples.iTuple()
    outputs: xtuples.iTuple = xtuples.iTuple()
    constraints: xtuples.iTuple = xtuples.iTuple()

    check_input = check_input
    add_input = add_input

    check_factor = check_factor
    add_factor = add_factor

    check_latent = check_latent
    add_latent = add_latent

    check_output = check_output
    add_output = add_output

    check_constraint = check_constraint
    add_constraint = add_constraint

    init_shape = init_shape
    init_params = init_params

    init_objective = init_objective
    init_apply = init_apply

    build = build_model
    optimise = optimise_model


# ---------------------------------------------------------------


# ---------------------------------------------------------------
