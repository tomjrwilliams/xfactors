from __future__ import annotations

import functools
import itertools

import typing
import dataclasses

import datetime

import numpy
import pandas

import pathlib

import scipy.stats

import plotly.express

import xtuples as xt

# ---------------------------------------------------------------

def test_nd():
    return numpy.array([
        [1., 0., -1.],
        [-1., 1., 1.],
        [-2., 2., 0.],
    ])


def test_nd_nan():
    return numpy.array([
        [numpy.NaN, 0., -1., -1, 1.],
        [-1., numpy.NaN, 1., -1, 1.],
        [-2., 2., numpy.NaN, -1, 1.],
    ])

# ---------------------------------------------------------------


def f_where_top(f, top):

    assert "n" in top, top
    n = top["n"]

    long = top.get("long", None)
    short = top.get("short", None)

    assert len([v for v in [long, short]]) > 0, top

    def f_res(vs):

        order = numpy.argsort(vs)
        res = numpy.zeros_like(vs)

        if long and len(order):
            i = order[-n:]
            w = f(vs[i]) * long
            res[i] = w

        if short and len(order):
            i = order[:n]
            # reverse order of signal (before we multiply by short)
            w = f(vs[i] * -1) * short
            res[i] = w

        return res

    return f_res

def f_where_not_na(nd, f=None, top = None, dp = None):
    """
    >>> f_where_not_na(test_nd_nan(), f = weights_equal, dp=2)
    array([[ nan, 0.25, 0.25, 0.25, 0.25],
           [0.25,  nan, 0.25, 0.25, 0.25],
           [0.25, 0.25,  nan, 0.25, 0.25]])
    >>> f_where_not_na(test_nd_nan(), f = weights_proportional, dp=2)
    array([[ nan, 0.33, 0.  , 0.  , 0.67],
           [0.  ,  nan, 0.5 , 0.  , 0.5 ],
           [0.  , 0.5 ,  nan, 0.12, 0.38]])
    >>> f_where_not_na(test_nd_nan(), f = weights_softmax, dp=2)
    array([[ nan, 0.22, 0.08, 0.08, 0.61],
           [0.06,  nan, 0.44, 0.06, 0.44],
           [0.01, 0.7 ,  nan, 0.03, 0.26]])
    >>> f_where_not_na(test_nd_nan(), f = weights_linear, dp=2)
    array([[ nan, 0.33, 0.67, 0.  , 1.  ],
           [0.  ,  nan, 0.67, 0.33, 1.  ],
           [0.  , 0.67,  nan, 1.  , 0.33]])
    >>> f_where_not_na(test_nd_nan(), f = weights_equal, top = dict(
    ...     n=1, long=1
    ... ), dp=2)
    array([[nan,  0.,  0.,  0.,  1.],
           [ 0., nan,  0.,  0.,  1.],
           [ 0.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_proportional, top = dict(
    ...     n=1, long=1
    ... ), dp=2)
    array([[nan,  0.,  0.,  0.,  1.],
           [ 0., nan,  0.,  0.,  1.],
           [ 0.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_softmax, top = dict(
    ...     n=1, long=1
    ... ), dp=2)
    array([[nan,  0.,  0.,  0.,  1.],
           [ 0., nan,  0.,  0.,  1.],
           [ 0.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_linear, top = dict(
    ...     n=1, long=1
    ... ), dp=2)
    array([[nan,  0.,  0.,  0.,  1.],
           [ 0., nan,  0.,  0.,  1.],
           [ 0.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_equal, top = dict(
    ...     n=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  0.],
           [-1., nan,  0.,  0.,  0.],
           [-1.,  0., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_proportional, top = dict(
    ...     n=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  0.],
           [-1., nan,  0.,  0.,  0.],
           [-1.,  0., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_softmax, top = dict(
    ...     n=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  0.],
           [-1., nan,  0.,  0.,  0.],
           [-1.,  0., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_linear, top = dict(
    ...     n=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  0.],
           [-1., nan,  0.,  0.,  0.],
           [-1.,  0., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_equal, top = dict(
    ...     n=1, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  1.],
           [-1., nan,  0.,  0.,  1.],
           [-1.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_proportional, top = dict(
    ...     n=1, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  1.],
           [-1., nan,  0.,  0.,  1.],
           [-1.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_softmax, top = dict(
    ...     n=1, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  1.],
           [-1., nan,  0.,  0.,  1.],
           [-1.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_linear, top = dict(
    ...     n=1, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -1.,  0.,  1.],
           [-1., nan,  0.,  0.,  1.],
           [-1.,  1., nan,  0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_equal, top = dict(
    ...     n=2, long=1, short=-1
    ... ), dp=2)
    array([[ nan,  0.5, -0.5, -0.5,  0.5],
           [-0.5,  nan,  0.5, -0.5,  0.5],
           [-0.5,  0.5,  nan, -0.5,  0.5]])
    >>> f_where_not_na(test_nd_nan(), f = weights_proportional, top = dict(
    ...     n=2, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., nan, nan,  1.],
           [nan, nan, nan, nan, nan],
           [-1.,  1., nan, -0.,  0.]])
    >>> f_where_not_na(test_nd_nan(), f = weights_softmax, top = dict(
    ...     n=2, long=1, short=-1
    ... ), dp=2)
    array([[  nan,  0.27, -0.5 , -0.5 ,  0.73],
           [-0.5 ,   nan,  0.5 , -0.5 ,  0.5 ],
           [-0.73,  0.73,   nan, -0.27,  0.27]])
    >>> f_where_not_na(test_nd_nan(), f = weights_linear, top = dict(
    ...     n=2, long=1, short=-1
    ... ), dp=2)
    array([[nan,  0., -0., -1.,  1.],
           [-0., nan,  0., -1.,  1.],
           [-1.,  1., nan, -0.,  0.]])
    """
    if top is not None:
        f = f_where_top(f, top)

    try:
        isnan = numpy.isnan(nd)
        notnan = numpy.logical_not(isnan)
    except:
        assert False, [nd.shape, type(nd)]
    
    if isnan.sum() == 0:
        return nd.apply_along_axis(f)
    
    not_nan_inds = xt.iTuple(notnan).map(
        lambda r: numpy.nonzero(r)
    )
    
    def row_res(vs, js):
        res = numpy.ones_like(nd[0]) * numpy.NaN 
        res[js] = vs
        return res

    res = numpy.vstack([
        numpy.ones_like(nd[0]) * numpy.NaN 
        if not len(i)
        else row_res(
            f(nd[r][i]),
            i,
        )
        for r, i in not_nan_inds.enumerate()
    ])
    if dp is not None:
        return numpy.round(res, dp)
    return res

# ---------------------------------------------------------------

def weights_equal(nd):
    """
    >>> weights_equal(test_nd()[0])
    array([0.33333333, 0.33333333, 0.33333333])
    >>> weights_equal(test_nd()[1])
    array([0.33333333, 0.33333333, 0.33333333])
    >>> weights_equal(test_nd()[2])
    array([0.33333333, 0.33333333, 0.33333333])
    """
    return numpy.ones_like(nd) / len(nd)

def weights_proportional(nd):
    """
    >>> weights_proportional(test_nd()[0])
    array([0.66666667, 0.33333333, 0.        ])
    >>> weights_proportional(test_nd()[1])
    array([0. , 0.5, 0.5])
    >>> weights_proportional(test_nd()[2])
    array([0.        , 0.66666667, 0.33333333])
    """
    v_min = numpy.min(nd)
    if len(nd) > 1:
        nd_shift = nd - v_min
    else:
        nd_shift = nd
    return numpy.divide(nd_shift, sum(nd_shift))

import scipy.special

def weights_softmax(nd):
    """
    >>> weights_softmax(test_nd()[0])
    array([0.66524096, 0.24472847, 0.09003057])
    >>> weights_softmax(test_nd()[1])
    array([0.06337894, 0.46831053, 0.46831053])
    >>> weights_softmax(test_nd()[2])
    array([0.01587624, 0.86681333, 0.11731043])
    """
    v_min = numpy.min(nd)
    if len(nd) > 1:
        nd_shift = nd - v_min
    else:
        nd_shift = nd
    return scipy.special.softmax(nd_shift)

def weights_linear(nd):
    """
    >>> weights_linear(test_nd()[0])
    array([1. , 0.5, 0. ])
    >>> weights_linear(test_nd()[1])
    array([0. , 0.5, 1. ])
    >>> weights_linear(test_nd()[2])
    array([0. , 1. , 0.5])
    """
    if len(nd) == 1:
        return numpy.ones(1)
    inds = numpy.argsort(nd)
    return numpy.linspace(0, 1, len(nd))[inds]

# ---------------------------------------------------------------

def calc_weights(
    signal_df,
    universe_df=None,
    equal=None,
    linear=None,
    proportional=None,
    softmax=None,
    given=None,
    inv=None,
    gross=None,
    top=None,
):
    index = signal_df.index

    if universe_df is not None:
        index = index.union(universe_df.index)

        signal_df = signal_df.reindex(index)
        universe_df = universe_df.reindex(index)

        universe_df = (
            universe_df.replace(0, numpy.NaN)
            .replace(False, numpy.NaN)
            #
        )

        for ticker in signal_df.columns:
            if ticker not in universe_df.columns:
                signal_df[ticker] = pandas.Series(
                    index=signal_df[ticker].index,
                    data=[
                        numpy.NaN for _ in index
                    ],
                )
            else:
                signal_df[ticker] *= universe_df[ticker]
            #
    
    # left with na where not in universe

    kws = dict(
        equal=equal,
        linear=linear,
        softmax=softmax,
        proportional=proportional,
        given=given,
    )
    assert len(list(v for v in kws.values() if v)) <= 1, kws

    nd = numpy.nan_to_num(
        signal_df.values,
        nan=numpy.NaN,
        posinf=numpy.NaN,
        neginf=numpy.NaN,
    ).astype(float)

    if inv:
        nd = numpy.nan_to_num(
            1 / nd,
            nan=numpy.NaN,
            posinf=numpy.NaN,
            neginf=numpy.NaN,
        )
    if gross:
        nd = numpy.abs(nd)

    if given:
        vs = nd
    elif equal:
        vs = f_where_not_na(nd, top = top, f= weights_equal)
    elif linear:
        vs = f_where_not_na(nd, top = top, f= weights_linear)
    elif softmax:
        vs = f_where_not_na(nd, top = top, f= weights_softmax)
    elif proportional:
        vs = f_where_not_na(nd, top = top, f= weights_proportional)
    else:
        assert False, kws
    
    return pandas.DataFrame(
        vs,
        index=signal_df.index,
        columns=signal_df.columns,
    )

# ---------------------------------------------------------------

def combine_dfs(
    signal_weights,
    weight_df=None,
    add=False,
    mul=False,
    norm=None,
):
    kws = dict(
        add=add,
        mul=mul,
    )
    assert len(list(v for v in kws.values() if v)) == 1, kws

    if weight_df is not None:
        signal_weights[None] = weight_df

    signal_ks = xt.iTuple.from_keys(signal_weights)

    signal_df = signal_weights[signal_ks[0]]

    for k in signal_ks[1:]:
        df = signal_weights[k]
        for col in signal_df.columns:
            if col in df:
                if add:
                    signal_df[col] = signal_df[col] + df[col]
                elif mul:
                    signal_df[col] = signal_df[col] * df[col]
                else:
                    assert False, kws
        for col in df.columns:
            if col not in signal_df:
                signal_df[col] = df[col]

    if norm is None:
        return signal_df

    else:
        assert isinstance(norm, dict), norm
    
    method = norm["method"]
    top = norm["top"]

    if method == "equal":
        return calc_weights(
            signal_df,
            equal=True,
            top=top,
        )
    elif method == "linear":
        return calc_weights(
            signal_df,
            linear=True,
            top=top,
        )
    elif method == "softmax":
        return calc_weights(
            signal_df,
            softmax=True,
            top=top,
        )
    elif method == "proportional":
        return calc_weights(
            signal_df,
            proportional=True,
            top=top,
        )
    else:
        assert False, norm

# ---------------------------------------------------------------

# gross defaults to 1.
# and then use the separate gross flag at strategy level for 
# gross up / down / neg

def ls_weights(
    signal_dfs,
    #
    weight_df=None,
    universe_df=None,
    # if truthy, either scalar
    # or None or dict
    #
    weight_kws = None,
    # equal=None,
    # linear=None,
    # softmax=None,
    # proportional=None,
    # top=None,
    # #
    # inv=None,
    # gross=None,
    #
    add=False,
    mul=False,
    norm=None,
    #
):
    if weight_kws is None:
        weight_kws = {}

    if not isinstance(signal_dfs, dict):
        signal_dfs = {None: signal_dfs}
        add = True

    # if isinstance(signal_dfs, dict):
    #     equal = {} if equal is None else equal
    #     linear = {} if linear is None else linear
    #     softmax = {} if softmax is None else softmax
    #     proportional = {} if proportional is None else proportional
    #     inv = {} if inv is None else inv
    #     gross = {} if gross is None else gross
    #     top = {} if top is None else top

    # weight_kws = {
    #     signal: dict(
    #         equal=(
    #             equal if not isinstance(equal, dict)
    #             else equal.get(signal, None)
    #         ),
    #         linear=(
    #             linear if not isinstance(linear, dict)
    #             else linear.get(signal, None)
    #         ),
    #         softmax=(
    #             softmax if not isinstance(softmax, dict)
    #             else softmax.get(signal, None)
    #         ),
    #         proportional=(
    #             proportional if not isinstance(proportional, dict)
    #             else proportional.get(signal, None)
    #         ),
    #         #
    #         inv=(
    #             inv if not isinstance(inv, dict)
    #             else inv.get(signal, None)
    #         ),
    #         gross=(
    #             gross if not isinstance(gross, dict)
    #             else gross.get(signal, None)
    #         ),
    #         top=(
    #             top if not isinstance(top, dict)
    #             else top.get(signal, None)
    #         ),
    #     )
    #     for signal in signal_dfs.keys()
    # }

    signal_weights = {
        signal: calc_weights(
            signal_dfs[signal],
            universe_df=universe_df,
            **weight_kws[signal],
        )
        for signal in weight_kws.keys()
    }
    if len(signal_weights.keys()) == 1:
        return signal_weights[list(signal_weights.keys())[0]]

    comb_kws = dict(
        add=add,
        mul=mul,
    )
    assert len(
        list(v for v in comb_kws.values() if v)
        #
    ) <= 1, comb_kws
    if not add and not mul:
        add = True

    return combine_dfs(
        signal_weights,
        weight_df=weight_df,
        add=add,
        mul=mul,
        norm=norm,
    )


# ---------------------------------------------------------------
