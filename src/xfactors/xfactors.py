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

NULL_SHAPE = "NULL_SHAPE"
NO_PARAMS = "NO_PARAMS"

def init_shape_null(self, model, data):
    return self

def init_params_null(obj, model, state):
    return obj, ()

def add_bindings(cls, methods, kfs):
    for k, f in kfs.items():
        if k == "init_shape" and f == "NULL_SHAPE":
            methods[k] = init_shape_null
        elif k == "init_params" and f == "NO_PARAMS":
            methods[k] = init_params_null
        else:
            methods[k] = f
    for k, f in methods.items():
        setattr(cls, k, f)
    if not hasattr(cls, "init_shape"):
        setattr(cls, "init_shape", init_shape_null)
    if not hasattr(cls, "init_params"):
        setattr(cls, "init_params", init_params_null)
    return cls

# ---------------------------------------------------------------

def check_location(loc):
    assert loc.domain in [None, 0, 1, 2], loc
    return True

PARAM = 0
RESULT = 1
CONSTRAINT = 2

# as we can follow paths through the model
def follow_path(path, acc):
    return xt.iTuple(path).fold(
        lambda acc, i: acc[i], initial=acc
    )

def get_location(loc, acc):
    return follow_path(loc.path, acc[loc.domain])

def f_follow_path(acc):
    def f(obj):
        if hasattr(obj, "domain"):
            return follow_path(obj.path, acc)
        return follow_path(obj, acc)
    return f

def f_get_location(acc):
    def f(loc):
        return get_location(loc, acc)
    return f

@xt.nTuple.decorate
class Location(typing.NamedTuple):

    domain: int # [0, 1, 2]
    path: xt.iTuple

    check = check_location
    access = get_location

    @classmethod
    def model(cls, *path):
        return cls(None, path)

    @classmethod
    def param(cls, *path):
        return cls(PARAM, path)

    @classmethod
    def result(cls, *path):
        return cls(RESULT, path)

    @classmethod
    def constraint(cls, *path):
        return cls(CONSTRAINT, path)

    def as_param(self):
        return Location(PARAM, self.path)

    def as_result(self):
        return Location(RESULT, self.path)

    def as_constraint(self):
        return Location(CONSTRAINT, self.path)

Loc = Location

# the model location shouldn't have domains?

# ---------------------------------------------------------------

class Stage(xt.iTuple):
    pass

def update_stage(model, i, stage):
    return model._replace(
        stages = (
            model.stages[:i].append(stage)
            .extend(model.stages[i+1:])
        )
    )

def add_stage(model, stage = None):
    return model._replace(
        stages = model.stages.append((
            stage if stage is not None else Stage()
        ))
    )

# 1 + n as we always have input stage pre-defined
def init_stages(model, n):
    return xt.iTuple.range(1 + n), model._replace(
        n_stages=n+1,
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
    return update_stage(
        model, 0, stage.append(obj._replace(
            loc=Location.model(0, stage.len())
        ))
    )

def input_bindings(**kfs):
    def decorator(cls):
        methods = dict(
            check = check_input,
            add = add_input,
        )
        return add_bindings(cls, methods, kfs)
    return decorator

# ---------------------------------------------------------------

def check_operator(obj, model):
    assert hasattr(obj, "loc"), obj
    assert hasattr(obj, "shape"), obj
    return True

def add_operator(model, i, obj):
    stage = model.stages[i]
    assert check_operator(obj, model)
    return update_stage(
        model, i, stage.append(
            obj._replace(
                loc=Location.model(i, stage.len())
            )
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

def constraint_bindings(init_params = NO_PARAMS, **kfs):
    def decorator(cls):
        methods = dict(
            check = check_constraint,
            add = add_constraint,
        )
        kfs["init_params"] = init_params
        return add_bindings(cls, methods, kfs)
    return decorator

# ---------------------------------------------------------------

def init_shapes(model, data):
    return model.stages.fold(
        lambda model, stage: model._replace(
            stages=model.stages.append(
                stage.map(
                    operator.methodcaller(
                        "init_shape", model, data
                    )
                )
            )
        ),
        initial=model._replace(stages=xt.iTuple()),
    )

def init_params(model, data):

    def f_acc(model, params, stage, data):
        stage, stage_params = stage.map(
            operator.methodcaller("init_params", model, data)
        ).zip().map(xt.iTuple)
        return model._replace(
            stages=model.stages.append(stage),
        ), params.append(stage_params)

    model, params = model.stages.fold(
        lambda model_params, stage: f_acc(
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

# TODO: might be value in multi step inputs?

# ie. for those calcs that only have to be run once?

# or, no, have a caching flag that says they're static, pure function of inputs

# keep the inputs as a separate first layer, that only depends on the values of the data
# can then have a flag for static operators that only depend on the value of the inputs

# and then the rest are dynamic, eg. value of params (which change during optimisation)

# init_results = just inputs (ie. parsed data)
# no model results yet
def init_objective(model, init_params, data):
    init_results = xt.iTuple().append(
        model.stages[0].map(
            operator.methodcaller("apply", (init_params, data,))
        )
    )
    def f(params, results):
        results = model.stages[1:].fold(
            lambda res, stage: res.append(stage.map(
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
            model.stages[0].map(
                operator.methodcaller("apply", (params, data,))
            )
        )
        results = model.stages[1:].fold(
            lambda res, stage: res.append(stage.map(
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
    if isinstance(v, (xt.iTuple, Stage)):
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

    params: xt.iTuple = xt.iTuple()
    stages: xt.iTuple = xt.iTuple.one(Stage())
    constraints: xt.iTuple = xt.iTuple()

    n_stages: int = None

    add_input = add_input
    add_stage = add_stage
    add_operator = add_operator
    add_constraint = add_constraint

    init_stages = init_stages

    init_shapes = init_shapes
    init_params = init_params

    init_objective = init_objective
    init_apply = init_apply

    build = build_model
    optimise = optimise_model


# ---------------------------------------------------------------
