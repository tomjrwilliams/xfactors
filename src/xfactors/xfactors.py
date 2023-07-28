import enum

import operator
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

import xtuples as xt

from . import rand
from . import dates

# ---------------------------------------------------------------

from jax.config import config 
# config.update("jax_debug_nans", True) 

# ---------------------------------------------------------------

def add_bindings(cls, methods, kfs):
    for k, f in kfs.items():
        methods[k] = f
    for k, f in methods.items():
        setattr(cls, k, f)
    return cls

# ---------------------------------------------------------------

def check_location(loc):
    assert loc.domain in [0, 1, 2], loc
    return True

PARAM = 0
RESULT = 1
CONSTRAINT = 2

# as we can follow paths through the model
def follow_path(path, acc):
    return xt.iTuple(path).fold(
        lambda acc, i: acc[i], initial=acc
    )

def f_follow_path(acc):
    def f(obj):
        if hasattr(obj, "domain"):
            return follow_path(obj.path, acc)
        return follow_path(obj, acc)
    return f

def get_location(loc, acc):
    if loc.domain == RESULT and isinstance(acc, Model):
        return follow_path(
            (loc.path[0], 0, *loc.path[1:]),
            acc[loc.domain]
        )
    return follow_path(loc.path, acc[loc.domain])

def f_get_location(acc):
    def f(loc):
        return get_location(loc, acc)
    return f

@xt.nTuple.decorate
class Location(typing.NamedTuple):

    domain: int # [0, 1]
    path: xt.iTuple

    check = check_location
    access = get_location

    @classmethod
    def param(cls, *path):
        return cls(PARAM, path)

    @classmethod
    def result(cls, *path):
        return cls(RESULT, path)

    @classmethod
    def constraint(cls, *path):
        return cls(CONSTRAINT, path)

Loc = Location

# ---------------------------------------------------------------

def update_stage(model, i, stage):
    return model._replace(
        stages = (
            model.stages[:i].append(stage)
            .extend(model.stages[i+1:])
        )
    )

# ---------------------------------------------------------------

def check_input(obj, model):
    assert hasattr(obj, "loc"), obj
    assert hasattr(obj, "shape"), obj
    return True

def add_input(model, obj):
    """
    Inputs are always the first stage.
    """
    assert check_input(obj, model)
    stage = model.stages[0]
    stage = stage._replace(
        operators=stage.operators.append(obj._replace(
            loc=Location.result(0, len(stage.operators))
        ))
    )
    return update_stage(model, 0, stage)

def input_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_input,
            add = add_input,
        )
        return add_bindings(cls, methods, kfs)
    return decorator

# ---------------------------------------------------------------

@input_bindings()
@xt.nTuple.decorate
class Input_DataFrame_Wide(typing.NamedTuple):

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        # path[0] = stage, so path[1] = index of data element
        return self._replace(
            shape=data[self.loc.path[1]].values.shape
        )

    def init_params(self, model, state):
        return self, ()
    
    def apply(self, state):
        params, data = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

@input_bindings()
@xt.nTuple.decorate
class Input_DataFrame_Tall(typing.NamedTuple):

    # fields to specify if keep index and ticker map

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        # path[0] = stage, so path[1] = index of data element
        return self._replace(
            shape=data[self.loc.path[1]].values.shape
        )

    def init_params(self, model, state):
        return self, ()
    
    def apply(self, state):
        params, data = state
        df = data[self.loc.path[-1]]
        return jax.numpy.array(df.values)

# ---------------------------------------------------------------

def check_stage(obj, model):
    assert hasattr(obj, "operators"), obj
    assert hasattr(obj, "loc"), obj
    # check dependency structure of all operators
    return True

def add_stage(model, obj = None):
    if obj is None:
        obj = Stage()
    check_stage(obj, model)
    return model._replace(
        stages = model.stages.append(obj._replace(
            loc=Location.result(len(model.stages))
        ))
        #
    )

def stage_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_stage,
            add = add_stage,
        )
        return add_bindings(cls, methods, kfs)
    return decorator

def check_operator(obj, stage, model):
    assert hasattr(obj, "loc"), obj
    assert hasattr(obj, "shape"), obj
    return True

def add_operator(model, i, obj):
    stage = model.stages[i]
    assert check_operator(obj, stage, model)
    return update_stage(
        model, i, stage._replace(
            operators = stage.operators.append(obj._replace(
                loc=Location.result(
                    *stage.loc.path, len(stage.operators)
                )
            ))
            #
        )
    )

def operator_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_operator,
            add = add_operator,
        )
        return add_bindings(cls, methods, kfs)
    return decorator

@stage_bindings()
@xt.nTuple.decorate
class Stage(typing.NamedTuple):

    operators: xt.iTuple = xt.iTuple()

    loc: Location = None

    @classmethod
    @property
    def INPUT(cls):
        return 0

    # commenting as confusing
    # @classmethod
    # def result(cls, stage):
    #     return stage + 1

def init_stages(n):
    return xt.iTuple.range(1 + n)

# ---------------------------------------------------------------

# pca eigen decomposition as an operator
# even though no optimisation is necessary?
# so for optimise, if no constraints, just return


@operator_bindings()
@xt.nTuple.decorate
class PCA(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        objs = self.sites.map(f_get_location(model))
        return self._replace(
            shape = (
                objs.map(lambda o: o.shape[1]).pipe(sum),
                self.n,
            )
        )

    def init_params(self, model, state):
        return self, ()

    def apply(self, state):
        data = jax.numpy.concatenate(
            self.sites.map(f_get_location(state)),
            axis=1,
        )
        eigvals, weights = jax.numpy.linalg.eig(jax.numpy.cov(
            jax.numpy.transpose(data)
        ))
        return eigvals, weights

@operator_bindings()
@xt.nTuple.decorate
class PCA_Encoder(typing.NamedTuple):
    
    n: int
    sites: xt.iTuple
    site: Location = None

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        objs = self.sites.map(f_get_location(model))
        return self._replace(
            shape = (
                objs.map(lambda o: o.shape[1]).pipe(sum),
                self.n,
            )
        )

    def init_params(self, model, state):
        if self.site is None:
            return self._replace(
                site=Location.param(*self.loc.path)
            ), rand.normal(self.shape)
        # TODO: check below, assumes weights generated elsewhere
        return self, rand.normal(self.shape)

    def apply(self, state):
        weights = get_location(self.site, state)
        return jax.numpy.matmul(
            jax.numpy.concatenate(
                self.sites.map(f_get_location(state)),
                axis=1,
            ),
            weights,
        )

Lin_Reg = PCA_Encoder

@operator_bindings()
@xt.nTuple.decorate
class PCA_Decoder(typing.NamedTuple):
    
    sites: xt.iTuple

    # TODO: generalise to sites_weight and sites_data
    # so that can spread across multiple prev stages
    # and then concat both, or if size = 1, then as below
    # can also pass as a nested tuple? probs cleaner to have separate

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        return self

    def init_params(self, model, state):
        return self, ()

    def apply(self, state):
        assert len(self.sites) == 2
        l_site, r_site = self.sites
        return jax.numpy.matmul(
            get_location(r_site, state), 
            get_location(l_site, state).T
        )

# TODO:

# PPCA EM E
# PPCA EM M

# ---------------------------------------------------------------

def check_latent(obj, model):
    assert obj.axis in [None, 0, 1]
    return check_operator(obj, model)

@operator_bindings(check = check_latent)
@xt.nTuple.decorate
class Latent(typing.NamedTuple):
    """
    axis: None = scalar, 0 = time series, 1 = ticker
    """

    axis: int
    # TODO init: collections.abc.Iterable = None

    loc: Location = None
    shape: xt.iTuple = None

    def init_shape(self, model, data):
        return self

    def init_params(self, model, params):
        axis = self.axis
        shape_latent = (
            (1,) if axis is None
            else (model.shape[axis], 1,) if axis == 0
            else (1, model.shape[axis],) if axis == 1
            else None
        )
        assert shape_latent is not None, self
        latent = rand.normal(shape_latent)
        return self, latent

# ---------------------------------------------------------------

def check_constraint(obj, model):
    return True

def add_constraint(model, obj):
    assert check_constraint(obj, model)
    return model._replace(
        constraints = model.constraints.append(obj._replace(
            loc=Location.constraint(model.constraints)
        ))
        #
    )

# def init_shape_constraint(obj, model, data):
#     return obj

def init_params_constraint(obj, model, state):
    return obj, ()

def constraint_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_constraint,
            add = add_constraint,
            # init_shape = init_shape_constraint,
            init_params = init_params_constraint,
        )
        return add_bindings(cls, methods, kfs)
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
@xt.nTuple.decorate
class Constraint_MSE(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        assert self.sites.len() == 2
        l_site, r_site = self.sites
        l = get_location(l_site, state)
        r = get_location(r_site, state)
        return loss_mse(l, r)

@constraint_bindings()
@xt.nTuple.decorate
class Constraint_Orthogonal(typing.NamedTuple):
    
    site: xt.iTuple

    loc: Location = None
    shape: xt.iTuple = None

    def apply(self, state):
        X = get_location(self.site, state)
        return loss_orthogonal(X)

@constraint_bindings()
@xt.nTuple.decorate
class Constraint_EigenVLike(typing.NamedTuple):
    
    sites: xt.iTuple

    loc: Location = None
    shape: xt.iTuple = None

    eigval_max: bool = True

    n_check: int = None

    def apply(self, state):
        assert len(self.sites) == 2
        w_site, f_site = self.sites
        w = get_location(w_site, state)
        f = get_location(f_site, state)
        cov = jax.numpy.cov(f.T)
        eigvals = jax.numpy.diag(cov)
        if self.n_check is not None:
            assert eigvals.shape[0] == self.n_check, (
                self, eigvals.shape,
            )
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

def init_shapes(model, data):
    return model.stages.fold(
        lambda model, stage: model._replace(
            stages=model.stages.append(stage._replace(
                operators = stage.operators.map(
                    operator.methodcaller(
                        "init_shape", model, data
                    )
                )
            ))
        ),
        initial=model._replace(stages=xt.iTuple()),
    )

def acc_init_params(model, params, stage, data):
    stage_operators, stage_params = stage.operators.map(
        operator.methodcaller("init_params", model, data)
    ).zip().map(xt.iTuple) # zip returns tuple not iTuple
    return model._replace(
        stages=model.stages.append(stage._replace(
            operators=stage_operators
        )),
    ), params.append(stage_params)

def init_params(model, data):
    model, params = model.stages.fold(
        lambda model_params, stage: acc_init_params(
            *model_params, stage, data
        ),
        initial=(
            model._replace(stages=xt.iTuple()), 
            xt.iTuple(),
            #
        )
    )
    return model, params

# ---------------------------------------------------------------

def init_objective(model, init_params, data):

    # init_results = just inputs (ie. parsed data)
    # no model results yet
    init_results = xt.iTuple().append(
        model.stages[0].operators.map(
            operator.methodcaller("apply", (init_params, data,))
        )
    )

    def f(params, results):
        results = model.stages[1:].fold(
            lambda res, stage: res.append(stage.operators.map(
                operator.methodcaller("apply", (params, res,))
            )),
            initial=xt.iTuple(results),
        )
        return jax.numpy.stack(
            model.constraints.map(
                operator.methodcaller("apply", (params, results,))
            ).pipe(list)
        ).sum()

    return init_results, f

def init_apply(model):
    def f(params, data, sites = None):
        # init_results = just inputs (ie. parsed data)
        # no model results yet
        init_results = xt.iTuple().append(
            model.stages[0].operators.map(
                operator.methodcaller("apply", (params, data,))
            )
        )
        results = model.stages[1:].fold(
            lambda res, stage: res.append(stage.operators.map(
                operator.methodcaller("apply", (params, res,))
            )),
            initial=init_results,
        )
        if sites is None:
            return results
        return {
            k: get_location(s, results)
            for k, s in sites.items()
        }
    return f

# ---------------------------------------------------------------

def build_model(model, data):
    model = model.init_shapes(data)
    model, params = model.init_params(data)
    results, objective = model.init_objective(params, data)
    apply = model.init_apply()
    return model, params, results, objective, apply

def to_tuple_rec(v):
    if isinstance(v, xt.iTuple):
        return v.map(to_tuple_rec).pipe(tuple)
    return v

def optimise_model(
    model, 
    params, 
    results, 
    objective, 
    lr = 0.01,
    iters=1000,
    verbose=True,
):
    if not model.constraints.len():
        return model, params
    
    opt = optax.adam(lr)
    solver = jaxopt.OptaxSolver(
        opt=opt, fun=objective, maxiter=iters
    )

    params = params.pipe(to_tuple_rec)
    results = results.pipe(to_tuple_rec)

    state = solver.init_state(params, results)

    for i in range(iters):
        params, state = solver.update(
            params,
            state,
            results,
        )
        if i % int(iters / 10) == 0 or i == iters - 1:
            if verbose: print(i, state.error)

        # TODO: early termination if error stops changing

    # TODO: for all operators (inputs / constraints) that
    # sepcify they need a key

    # generate a key map, with the same shape as state
    # that can be indexed into (expand the function sig to then be:
    # (state, keys)

    return model, params

# ---------------------------------------------------------------

@xt.nTuple.decorate
class Model(typing.NamedTuple):

    null: xt.iTuple = xt.iTuple()
    stages: xt.iTuple = xt.iTuple.one(Stage())
    constraints: xt.iTuple = xt.iTuple()

    add_input = add_input
    add_stage = add_stage
    add_operator = add_operator
    add_constraint = add_constraint

    init_shapes = init_shapes
    init_params = init_params

    init_objective = init_objective
    init_apply = init_apply

    build = build_model
    optimise = optimise_model


# ---------------------------------------------------------------
