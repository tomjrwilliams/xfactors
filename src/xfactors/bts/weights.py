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

# ---------------------------------------------------------------

def weights_equal(rs):
    isnan = numpy.isnan(rs)
    notnan = numpy.logical_not(isnan)
    mask = notnan.replace(False, numpy.NaN)
    counts = notnan.sum(axis=1)
    counts = numpy.expand_dims(counts, axis=1)
    ones = numpy.ones(rs.shape)
    weights = numpy.divide(ones, counts)
    return numpy.multiply(mask, weights)

def weights_proportional(rs):
    isnan = numpy.isnan(rs)
    notnan = numpy.logical_not(isnan)
    mask = notnan.replace(False, numpy.NaN)
    sums = numpy.expand_dims(
        rs.nan_to_num(0).sum(axis=1), axis = 1
    )
    return numpy.multiply(
        numpy.divide(rs, sums),
        mask,
    )

import scipy.special
def weights_softmax(rs):
    isnan = numpy.isnan(rs)
    notnan = numpy.logical_not(isnan)
    mask = notnan.replace(False, numpy.NaN)
    if isnan.sum() == 0:
        return rs.apply_along_axis(scipy.special.softmax)
    inds = {i: [] for i in range(len(rs))}
    inds = {
        **inds,
        **{
            g[0][0]: g for g in core.Array(
                list(map(tuple, numpy.argwhere(notnan)))
            ).groupBy(lambda ii: ii[0])
        }
    }
    width = len(rs[0])
    return numpy.array([
        [
            numpy.NaN for j in range(width)
        ]
        if not len(vs[i])
        else (
            lambda vs, js, jmap: [
                vs[jmap[j]] if j in js else numpy.NaN
                for j in range(width)
            ]
        )(
            scipy.special.softmax(
                rs[i][j] for j in inds[i]
            ),
            inds[i],
            {
                j: ii for ii, j in enumerate(inds[i])
            }
        )
        for i in inds
    ])

def weights_linear(rs):
    isnan = numpy.isnan(rs)
    notnan = numpy.logical_not(isnan)
    mask = notnan.replace(False, numpy.NaN)
    counts = notnan.sum(axis=1)
    weight_cache = {
        l: (
            lambda vs: vs / sum(vs)
        )(numpy.linspace(0, 1, num=l))
        for l in set(counts)
    }
    if isnan.sum() == 0:
        ws = weight_cache[len(rs[0])]
        return numpy.array([
            ws[r.argsort()]
            for r in rs
        ])
    inds = {i: [] for i in range(len(rs))}
    inds = {
        **inds,
        **{
            g[0][0]: g for g in core.Array(
                list(map(tuple, numpy.argwhere(notnan)))
            ).groupBy(lambda ii: ii[0])
        }
    }
    width = len(rs[0])
    return numpy.array([
        [
            numpy.NaN for j in range(width)
        ]
        if not len(vs[i])
        else (
            lambda vs, js, jmap: [
                vs[jmap[j]] if j in js else numpy.NaN
                for j in range(width)
            ]
        )(
            weight_cache[counts[i]][
                (rs[i][j] for j in inds[i]).argsort()
            ],
            inds[i],
            {
                j: ii for ii, j in enumerate(inds[i])
            }
        )
        for i in inds
    ])


def calc_weights(
    signal_df,
    universe_df=None,
    equal=None,
    linear=None,
    proportional=None,
    softmax=None,
    inv=None,
    gross=None,
):
    index = signal_df.index

    if universe_df is not None:
        index = index.union(universe_df)

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
                    signal_df[ticker].index,
                    [
                        numpy.NaN for _ in index
                    ],
                )
            else:
                signal_df[ticker] = (
                    signal_df[ticker] * universe_df[ticker]
                )
            #
    
    kws = dict(
        equal=equal,
        linear=linear,
        softmax=softmax,
        proportional=proportional,
    )
    assert len(v for v in kws.values() if v) <= 1, kws

    rs = signal_df[signal_df.columns].to_numpy()

    if inv:
        rs = (1 / inv).nan_to_num(
            nan=numpy.NaN,
            posinf=numpy.NaN,
            neginf=numpy.NaN,
        )
    if gross:
        rs = numpy.abs(rs)

    if equal:
        vs = weights_equal(rs) 
    elif linear:
        vs = weights_linear(rs)
    elif softmax:
        vs = weights_softmax(rs)
    elif proportional:
        vs = weights_proportional(rs)
    else:
        assert False, kws

    vs = vs.T
    
    return pandas.DataFrame({
        column: vs[i]
        for i, column in enumerate(signal_df.columns)
    })

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
    assert len(v for v in kws.values() if v) == 1, kws

    if weight_df is not None:
        signal_weights[None] = weight_df

    signal_ks = core.Array.from_keys(signal_weights)

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
    
    elif norm == "equal":
        return calc_weights(
            signal_df,
            equal=True,
        )
    elif norm == "linear":
        return calc_weights(
            signal_df,
            linear=True,
        )
    elif norm == "softmax":
        return calc_weights(
            signal_df,
            softmax=True,
        )
    elif norm == "proportional":
        return calc_weights(
            signal_df,
            proportional=True,
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
    equal=None,
    linear=None,
    softmax=None,
    proportional=None,
    #
    inv=None,
    gross=None,
    #
    add=False,
    mul=False,
    norm=None,
    #
):

    if not isinstance(signal_dfs, dict):
        signal_dfs = {None: signal_dfs}
        add = True

    if isinstance(signal_dfs, dict):
        equal = {} if equal is None else equal
        linear = {} if linear is None else linear
        softmax = {} if softmax is None else softmax
        proportional = {} if proportional is None else proportional
        inv = {} if inv is None else inv
        gross = {} if gross is None else gross

    weight_kws = {
        signal: dict(
            equal=(
                equal if not isinstance(equal, dict)
                else equal.get(signal, None)
            ),
            linear=(
                linear if not isinstance(linear, dict)
                else linear.get(signal, None)
            ),
            softmax=(
                softmax if not isinstance(softmax, dict)
                else softmax.get(signal, None)
            ),
            proportional=(
                proportional if not isinstance(proportional, dict)
                else proportional.get(signal, None)
            ),
            #
            inv=(
                inv if not isinstance(inv, dict)
                else inv.get(signal, None)
            ),
            gross=(
                gross if not isinstance(gross, dict)
                else gross.get(signal, None)
            ),
        )
        for signal in signal_dfs.keys()
    }

    signal_weights = {
        signal: calc_weights(
            signal_dfs[signal],
            universe_df=universe_df,
            **weight_kws[signal],
        )
        for signal in weight_kws.keys()
    }

    comb_kws = dict(
        add=add,
        mul=mul,
    )
    assert len(v for v in comb_kws.values() if v) <= 1, comb_kws
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
