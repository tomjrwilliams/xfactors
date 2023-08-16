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

from . import backtests
from . import signals
from . import weights

# ---------------------------------------------------------------

# use long / short to indicate if long only, short only
# or if one should be a factor strategy

# TODO: rename equity_ls_signal (as in, signal df vs weight df)
# as opposed to eg. factor_ls (below)

# 

def long_short_trend(
    df_returns,
    dfs_indices,
    universe_name,
    alpha=2 / 30,
    z = False,
    top_n=None,
    shift="2D",
    flip=False,
    kwargs=None,
    combine=True,
    signal_name="signal",
    f_signal=None,
    frequency=bt.algos.RunDaily(),
    **strat_kwargs,
):
    #
    if kwargs is None:
        kwargs = {}
    #

    # TODO: if f_signal is iterable
    # fold through, res = f_signal(prev)
    # signal name = name of first
    # fs below are then f_signal[1:]

    df_signal = signals.df_ewm(
        df_returns,
        alpha=alpha,
        z=z,
    )
    if flip:
        df_signal = df_signal * -1
    df_signal.index = utils.dates.date_index(
        df_signal.index.values
    )
    df_signal = utils.dfs.shift(df_signal, shift)
    if f_signal is not None:
        df_signal = pandas.DataFrame(
            f_signal(df_signal.values),
            columns=df_signal.columns,
            index=df_signal.index,
        )
    acc = {
        signal_name: df_signal,
    }
    #
    if top_n is not None:
        strat_kwargs["top_n"] = top_n
        if "top_n_df_name" not in strat_kwargs:
            strat_kwargs["top_n_df_name"] = signal_name
    #
    strat_kwargs["frequency"] = frequency
    return backtests.long_short(
        acc,
        dfs_indices,
        name="ewma({})({}, {}, z={}{})".format(
            round(alpha, 3),
            universe_name,
            type(frequency).__name__,
            (z if not isinstance(z, float) else round(z, 3)), 
            (
                "" if f_signal is None else ", f={}".format(
                    f_signal.__name__
                )
            )
        ),
        strat_kwargs=strat_kwargs,
        combine=combine,
        **kwargs
    )

# NOTE: can pass kwargs = dict(long | short =Strategy())
# to do trend singles vs eg. factor strategy

# or replace the above with just passing a weights_df
# calculated with the relevant weighting?

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
