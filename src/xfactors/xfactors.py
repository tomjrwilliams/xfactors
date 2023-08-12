from __future__ import annotations

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
import optax

import xtuples as xt

from . import utils

# ---------------------------------------------------------------

expand_dims = utils.shapes.expand_dims
expand_dims_like = utils.shapes.expand_dims_like

# ---------------------------------------------------------------

def check_location(loc):
    assert loc.domain in [None, 0, 1, 2], loc
    return True

PARAM = 0
RESULT = 1
CONSTRAINT = 2
RANDOM = 3

# ---------------------------------------------------------------

# as we can follow paths through the model
def follow_path(path, acc):
    return xt.iTuple(path).fold(
        lambda acc, i: acc[i], initial=acc
    )

def get_location(loc: typing.Optional[Location], acc):
    assert loc is not None
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
    def f(loc: typing.Optional[Location]):
        return get_location(loc, acc)
    return f

# def concatenate_sites(sites, state, **kws):
#     if len(sites) == 1:
#         return get_location(*sites, state)
#     return jax.numpy.concatenate(
#         sites.map(f_get_location(state)),
#         **kws,
#     )
    
# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Location(typing.NamedTuple):

    domain: typing.Optional[int] # [0, 1, 2]
    path: xt.iTuple

    # ---
    
    def check(self, *args, **kwargs):
        return check_location(self, *args, **kwargs)
        
    def access(self, *args, **kwargs):
        return get_location(self, *args, **kwargs)

    # ---

    @classmethod
    def model(cls, *path):
        return cls(None, xt.iTuple(path))

    @classmethod
    def param(cls, *path):
        return cls(PARAM, xt.iTuple(path))

    @classmethod
    def result(cls, *path):
        return cls(RESULT, xt.iTuple(path))

    @classmethod
    def constraint(cls, *path):
        return cls(CONSTRAINT, xt.iTuple(path))

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

OptionalLocation = typing.Optional[Location]

# ---------------------------------------------------------------

import abc

class Node(typing.Protocol):

    @abc.abstractmethod
    def init(
        self: NodeClass,
        site: Site,
        model: Model,
        data: tuple,
    ) -> tuple[NodeClass, tuple, tuple]:
        ...

    @abc.abstractmethod
    def apply(
        self: NodeClass,
        site: Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        ...

# TODO: could do auto shape checking fairly easily? optionally presumably

NodeClass = typing.TypeVar("NodeClass", bound=Node)

def init_null(
    self, site: Site, model: Model, data: tuple
) -> tuple[Node, tuple, tuple]:
    return self, (), ()

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Site(typing.NamedTuple):
    
    node: Node

    loc: typing.Optional[Location] = None
    shape: typing.Optional[xt.iTuple] = None

    def init(self, model: Model, data: tuple):
        node, shape, params = self.node.init(self, model, data)
        return self._replace(
            node=node,
            shape=xt.iTuple(shape),
        ), params

    def apply(
        self,
        site: Site,
        state: tuple
    ) -> typing.Union[tuple, jax.numpy.ndarray]:
        return self.node.apply(self, state)

    def access(self, state: tuple):
        assert self.loc is not None
        return self.loc.access(state)

OptionalSite = typing.Optional[Site]

# ---------------------------------------------------------------

class Stage(xt.iTuple):
    pass

def update_stage(
    model: Model,
    i: int,
    stage: Stage
) -> Model:
    return model._replace(
        stages = (
            model.stages[:i].append(stage)
            .extend(model.stages[i+1:])
        )
    )

def add_stage(
    model: Model,
    stage: typing.Optional[Stage] = None
) -> Model:
    """
    >>> Model().add_stage()
    Model(params=iTuple(), stages=iTuple(Stage(), Stage()), constraints=iTuple(), n_stages=None)
    """
    return model._replace(
        stages = model.stages.append((
            stage if stage is not None else Stage()
        ))
    )

# return 1 + n indices as we always have input stage pre-defined
def init_stages(
    model: Model, n: int
) -> tuple[Model, xt.iTuple]:
    return xt.iTuple.range(n).map(lambda i: None).fold(
        add_stage, initial=model
    ), xt.iTuple.range(n + 1)
    
# ---------------------------------------------------------------

def check_input(node: Node, model: Model) -> bool:
    assert hasattr(node, "loc"), node
    assert hasattr(node, "shape"), node
    return True

def add_input(model: Model, node: Node) -> Model:
    """
    Inputs are always the first stage.
    """
    assert check_input(node, model)
    stage = model.stages[0]
    return update_stage(
        model, 0, stage.append(Site(
            node,
            loc=Location.model(0, stage.len())
        ))
    )

def check_node(node: Node, model: Model) -> bool:
    assert hasattr(node, "loc"), node
    assert hasattr(node, "shape"), node
    return True

def add_node(model: Model, i: int, node: Node) -> Model:
    stage = model.stages[i]
    assert check_node(node, model)
    return update_stage(
        model, i, stage.append(Site(
            node,
            loc=Location.model(i, stage.len())
        ))
    )

def check_constraint(node: Node, model: Model) -> bool:
    return True

def add_constraint(model: Model, node: Node) -> Model:
    assert check_constraint(node, model)
    return model._replace(
        constraints = model.constraints.append(Site(
            node,
            loc=Location.constraint(model.constraints)
        ))
        #
    )

# ---------------------------------------------------------------

def init_model(model: Model, data: tuple) -> Model:

    def f_acc(model: Model, params, stage, data):
        stage, stage_params = stage.map(
            operator.methodcaller("init", model, data)
        ).zip().map(xt.iTuple)
        return model._replace(
            stages=model.stages.append(stage),
        ), params.append(stage_params)

    model, params = model.stages.fold(
        lambda model_params, stage: f_acc(
            model_params[0], model_params[1], stage, data
        ),
        initial=(
            model._replace(stages=xt.iTuple()), 
            xt.iTuple(),
            #
        )
    )
    return model._replace(params=params)

# ---------------------------------------------------------------

def init_objective(
    model: Model,
    data,
    rand_keys,
    jit = True,
    **flags,
):
    init_params = model.params
    init_results = xt.iTuple().append(
        model.stages[0].map(
            operator.methodcaller(
                "apply", (init_params, data, (), rand_keys, flags))
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
                "apply", (init_params, res, (), rand_keys, flags)
            )
        )),
        initial=init_results,
    )
    def f(params, rand_keys, **flags):
        results = model.stages[1 + n_static:].fold(
            lambda res, stage: res.append(stage.map(
                operator.methodcaller(
                    "apply", (params, res, (), rand_keys, flags)
                )
            )),
            initial=xt.iTuple(init_results),
        )
        loss = jax.numpy.stack(
            model.constraints.map(
                operator.methodcaller(
                    "apply", (params, results, (), rand_keys, flags)
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
    sites = None,
    **flags,
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
                "apply", (params, data, (), rand_keys, flags)
            )
        )
    )
    results = model.stages[1:].fold(
        lambda res, stage: res.append(stage.map(
            operator.methodcaller(
                "apply", (params, res, (), rand_keys, flags)
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

def gen_rand_keys(model: Model):
    ks = model.stages.map(
        lambda stage: stage.map(
            lambda o: (
                utils.rand.next_key() if o.random else None
            )
        )
    )
    n_keys = sum(ks.map(lambda sks: sks.filter(
        lambda v: v is not None
    ).len()))
    return ks.pipe(to_tuple_rec), n_keys

# ---------------------------------------------------------------

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

    rand_keys, _ = gen_rand_keys(model)

    objective = init_objective(
        model,
        data,
        rand_keys=rand_keys,
        jit = jit,
    )
    f_grad = jax.value_and_grad(objective)

    tries = 0
    for iter in range(rand_init + 1):
        
        _params = (
            model.params
            if params is None
            else model.init_params(data).params
        ).pipe(to_tuple_rec)

        _test_loss, test_grad = f_grad(
            _params, rand_keys,
        )

        try:
            assert not has_nan(test_grad), (test_loss, test_grad,)
            if test_loss is None or _test_loss < test_loss:
                test_loss = _test_loss
                params = _params

            tries += 1
        except:
            if iter == rand_init:
                assert tries > 0

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

    if max_error_unchanged is not None and max_error_unchanged < 1:
        max_error_unchanged *= iters

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

    model = model._replace(params=params)
    return model

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Model(typing.NamedTuple):

    params: xt.iTuple = xt.iTuple()
    stages: xt.iTuple = xt.iTuple.one(Stage())
    constraints: xt.iTuple = xt.iTuple()

    n_stages: typing.Optional[int] = None

    # TODO: render method
    # that renders the graph, with labelled boxs on stages

    # can use __name__ for node labelling

    # ---
    
    def add_input(self: Model, node: Node) -> Model:
        return add_input(self, node)

    def add_stage(
        self: Model,
        stage: typing.Optional[Stage] = None
    ) -> Model:
        return add_stage(self, stage=stage)

    def add_node(self: Model, i: int, node: Node) -> Model:
        return add_node(self, i, node)

    def add_constraint(self: Model, node: Node) -> Model:
        return add_constraint(self, node)

    # ---

    def init_stages(self: Model, n: int) -> tuple[Model, xt.iTuple]:
        return init_stages(self, n)

    def init(self: Model, data: tuple) -> Model:
        return init_model(self, data)

    # ---

    def init_objective(self, *args, **kwargs):
        return init_objective(self, *args, **kwargs)

    def optimise(self, *args, **kwargs) -> Model:
        return optimise_model(self, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return apply_model(self, *args, **kwargs)

    # ---

# ---------------------------------------------------------------
