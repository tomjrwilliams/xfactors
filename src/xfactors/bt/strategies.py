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

import ffn
import bt

import xtuples as xt
# import xtenors

from .. import utils
from .. import visuals

from . import backtests
from . import signals
from . import weights

# ---------------------------------------------------------------

def ls_equity_signal(
    acc,
    df_returns,
    universe_df,
    fs,
    kws,
    universe_name,
    shift="2D",
    flip=False,
    signal_name="signal",
    frequency=bt.algos.RunDaily(),
    ls_kwargs=None,
    strat_kwargs=None,
):
    
    if ls_kwargs is None:
        ls_kwargs = {}
    if strat_kwargs is None:
        strat_kwargs = {}
    
    assert fs.len() == kws.len()

    df_signal = fs.zip(kws).foldstar(
        lambda acc, f, _kws: f(acc, **_kws),
        initial=df_returns,
    )

    if flip:
        df_signal = df_signal * -1
    df_signal.index = utils.dates.date_index(
        df_signal.index.values
    )
    df_signal = utils.dfs.shift(df_signal, shift)
    
    assert signal_name not in acc, dict(
        signal_name=signal_name,
        keys=list(acc.keys())
    )
    acc[signal_name] = df_signal
    
    if "top_n" in strat_kwargs:
        if "top_n_df_name" not in strat_kwargs:
            strat_kwargs["top_n_df_name"] = signal_name
    
    strat_kwargs["frequency"] = frequency

    return backtests.long_short(
        acc,
        universe_df,
        name="{}({})".format(
            fs.zip(kws).foldstar(
                lambda acc, f, _kws: "{}({}{})".format(
                    f.__name__, 
                    visuals.formatting.kwargs(_kws),
                    acc if acc == "" else ", " + acc,
                ),
                initial="",
            ),
            visuals.formatting.args([
                universe_name,
                type(frequency).__name__,
                (
                    visuals.formatting.args(
                        fs[1:].map(lambda f: f.__name__)
                    )
                ),
                visuals.formatting.kwargs(ls_kwargs),
            ])
        ),
        signal_name=signal_name,
        strat_kwargs=strat_kwargs,
        **ls_kwargs
    )

def ls_equity_weights(
    acc,
    signal_dfs,
    weight_kws,
    universe_df,
    universe_name,
    combine_kws=None,
    shift="2D",
    flip=False,
    weights_name="weights",
    frequency=bt.algos.RunDaily(),
    **kwargs,
):
    
    df_weights = weights.ls_weights(
        signal_dfs,
        universe_df=universe_df,
        weight_kws=weight_kws,
        **({} if combine_kws is None else combine_kws),
    )

    if flip:
        df_weights = df_weights * -1

    df_weights.index = utils.dates.date_index(
        df_weights.index.values
    )
    df_weights = utils.dfs.shift(df_weights, shift)

    assert weights_name not in acc, dict(
        weights_name=weights_name,
        keys=list(acc.keys())
    )
    acc[weights_name] = df_weights
    
    kwargs["frequency"] = frequency

    return backtests.build(
        acc,
        universe_df,
        name="combine({}, {})".format(
            visuals.formatting.kwargs(combine_kws),
            visuals.formatting.args([universe_name] + [
                "weights({})".format(
                    visuals.formatting.kwargs(dict(
                        signal=signal_k,
                        **weight_kws[signal_k]
                    ))
                )
                for signal_k in signal_dfs.keys()
            ]),
        ),
        weights_df_name=weights_name,
        **kwargs
    )

# ---------------------------------------------------------------

def factor_ls(

):

    # strategy per factor
    # rolling weight by dataframe of loadings

    # then a signal dataframe of the factor return trend

    # and then a combination allocating long short
    # amongst the factors

    return

def markov_factor_ls(
    
):

    # strategy per factor
    # rolling weight by dataframe of loadings

    # then a signal dataframe of the factor return trend
    # of current weights, or rolling weights?

    # and then a combination allocating long short
    # amongst the factors?

    return

# ---------------------------------------------------------------
