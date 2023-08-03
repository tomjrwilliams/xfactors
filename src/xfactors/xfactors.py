import enum

import operator
import collections
# import collections.abc

import functools
import itertools
from re import A

import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import jaxopt
import jaxopt.perturbations
import optax

import xtuples as xt

from . import rand
from . import dates


# ---------------------------------------------------------------

def expand_dims(v, axis, size):
    v_expand = jax.numpy.expand_dims(v, axis)
    if axis == -1:
        axis = len(v_expand.shape) - 1
    res = jax.numpy.tile(
        v_expand,
        tuple([
            *[1 for _ in v.shape[:axis]],
            size,
            *[1 for _ in v.shape[axis:]],
        ])
    )
    return res

def expand_dims_like(v, axis, like):
    v_expand = jax.numpy.expand_dims(v, axis)
    if axis == -1:
        axis = len(v_expand.shape) - 1
    return jax.numpy.tile(
        v_expand,
        tuple([
            *[1 for _ in v.shape[:axis]],
            like.shape[axis],
            *[1 for _ in v.shape[axis:]],
        ])
    )
# ---------------------------------------------------------------

def return_false(self):
    return False

def return_true(self):
    return True

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
    if not hasattr(cls, "static"):
        setattr(cls, "static", property(return_false))
    if not hasattr(cls, "random"):
        setattr(cls, "random", property(return_false))
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
RANDOM = 3

# as we can follow paths through the model
def follow_path(path, acc):
    return xt.iTuple(path).fold(
        lambda acc, i: acc[i], initial=acc
    )

def get_location(loc, acc):
    try:
        return follow_path(loc.path, acc[(
            RESULT if loc.domain is None else loc.domain
        )])
    except:
        assert False, loc

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

def concatenate_sites(sites, state, **kws):
    if len(sites) == 1:
        return get_location(*sites, state)
    return jax.numpy.concatenate(
        sites.map(f_get_location(state)),
        **kws,
    )

# ---------------------------------------------------------------

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

    def as_random(self):
        return Location(RANDOM, self.path)

    def as_model(self):
        return Location(None, self.path)

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
    return xt.iTuple.range(n).map(lambda i: None).fold(
        add_stage, initial=model
    ), xt.iTuple.range(n + 1)
    
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
    return model._replace(params=params)

# ---------------------------------------------------------------

def init_objective(model, data, rand_keys, jit = True):
    init_params = model.params
    init_results = xt.iTuple().append(
        model.stages[0].map(
            operator.methodcaller(
                "apply", (init_params, data, (), rand_keys))
        )
    )
    n_static = model.stages[1:].len_range().last_where(
        lambda i: model.stages[1 + i].all(lambda o: o.static),
    )
    n_static = 0 if n_static is None else n_static + 1
    # TODO: raise warning if in stage, some static but not all 
    # or static found in stage after last all static stage
    init_results = model.stages[1: 1 + n_static].fold(
        lambda res, stage: res.append(stage.map(
            operator.methodcaller(
                "apply", (init_params, res, (), rand_keys)
            )
        )),
        initial=init_results,
    )
    def f(params, rand_keys):
        results = model.stages[1 + n_static:].fold(
            lambda res, stage: res.append(stage.map(
                operator.methodcaller(
                    "apply", (params, res, (), rand_keys)
                )
            )),
            initial=xt.iTuple(init_results),
        )
        loss = jax.numpy.stack(
            model.constraints.map(
                operator.methodcaller(
                    "apply", (params, results, (), rand_keys)
                )
            ).pipe(list)
        ).sum()
        return loss
    if jit:
        return jax.jit(f)
    return f

def apply_model(
    model,
    data,
    rand_keys=None,
    params = None,
    sites = None
):
    if params is None:
        params = model.params

    if rand_keys is None:
        rand_keys, _ = gen_rand_keys(model)

    # init_results = just inputs (ie. parsed data)
    # no model results yet
    init_results = xt.iTuple().append(
        model.stages[0].map(
            operator.methodcaller(
                "apply", (params, data, (), rand_keys)
            )
        )
    )
    results = model.stages[1:].fold(
        lambda res, stage: res.append(stage.map(
            operator.methodcaller(
                "apply", (params, res, (), rand_keys)
            )
        )),
        initial=init_results,
    )
    if sites is None:
        return results
    return {
        k: get_location(s, results)
        for k, s in sites.items()
    }

# ---------------------------------------------------------------

def gen_rand_keys(model):
    ks = model.stages.map(
        lambda stage: stage.map(
            lambda o: (
                rand.next_key() if o.random else None
            )
        )
    )
    n_keys = sum(ks.map(lambda sks: sks.filter(
        lambda v: v is not None
    ).len()))
    return ks.pipe(to_tuple_rec), n_keys

# ---------------------------------------------------------------

def init_shapes_params(model, data):
    model = (
        model.init_shapes(data)
        .init_params(data)
    )
    return model

def to_tuple_rec(v):
    if isinstance(v, (xt.iTuple, Stage)):
        return v.map(to_tuple_rec).pipe(tuple)
    return v

def has_nan(v):
    if isinstance(v, tuple):
        return any([has_nan(vv) for vv in v])
    return numpy.isnan(v).any()

def init_optimisation(
    model,
    data,
    jit=True,
    rand_init=0,
):
    
    test_loss = None
    params = objective = None

    for _ in range(rand_init + 1):
        
        _params = (
            model.params
            if params is None
            else model.init_params(data).params
        ).pipe(to_tuple_rec)

        rand_keys, _ = gen_rand_keys(model)

        _objective = init_objective(
            model._replace(params=_params),
            data,
            rand_keys=rand_keys,
            jit = jit,
        )

        f_grad = jax.value_and_grad(_objective)
        _test_loss, test_grad = f_grad(
            _params, rand_keys,
        )

        assert not has_nan(test_grad), (test_loss, test_grad,)

        if test_loss is None or _test_loss < test_loss:
            test_loss = _test_loss
            params = _params
            objective = _objective

    return params, objective

def optimise_model(
    model, 
    data,
    iters=1000,
    verbose=True,
    jit=True,
    rand_init=0,
    max_error_unchanged=None,
    lr = 0.01,
    opt=None,
):

    if not model.constraints.len():
        return model
    
    params, objective = init_optimisation(
        model,
        data,
        rand_init=rand_init,
        jit=jit,
    )

    if opt is None:
        opt = optax.adam(lr)
    
    # NOTE: use optax.sgd(1) for em algos

    solver = jaxopt.OptaxSolver(
        opt=opt, fun=objective, maxiter=iters, jit=jit
    )
    state = solver.init_state(params) 

    error = None
    params_opt = None

    error_min = None
    since_min = 0

    rand_keys, n_random = gen_rand_keys(model)

    for i in range(iters):
        params, state = solver.update(
            params,
            state,
            rand_keys,
        )
        error = state.error

        if i % int(iters / 10) == 0 or i == iters - 1:
            if verbose: print(i, error)
        
        if error_min is None or error < error_min:
            error_min = error
            since_min = 0
            params_opt = params
        else:
            since_min += 1

        if (
            max_error_unchanged is not None 
            and since_min >= max_error_unchanged
        ):
            params = params_opt
            break

        if n_random > 0:
            rand_keys, _ = gen_rand_keys(model)

    # TODO: for all operators (inputs / constraints) that
    # sepcify they need a key

    # generate a key map, with the same shape as state
    # that can be indexed into (expand the function sig to then be:
    # (state, keys)

    model = model._replace(params=params)
    return model

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
    init_shapes_params = init_shapes_params

    init_objective = init_objective

    optimise = optimise_model

    apply = apply_model


# ---------------------------------------------------------------
