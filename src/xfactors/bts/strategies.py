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

# ---------------------------------------------------------------

def weight_algos(
    weights=None,
    # scalars
    equal=False,
    weight_df_name=None,
    # kwargs
    inv_vol=None,
    erc=None,
    mean_var=None,
    target_vol=None,
    #
    weight_limit=None,
):
    algos = []

    weight_kwargs = dict(
        weights=weights,
        #
        equal=equal,
        weight_df_name=weight_df_name,
        #
        inv_vol=inv_vol,
        erc=erc,
        mean_var=mean_var,
        target_vol=target_vol,
    )
    assert len(list(
        v for v in weight_kwargs.values() if v
        #
    )) == 1, weight_kwargs

    if weights:
        algos.append(bt.algos.WeighSpecified(**weights))

    elif equal:
        algos.append(bt.algos.WeighEqually())

    elif weight_df_name:
        algos.append(bt.algos.WeighTarget(weight_df_name))

    elif inv_vol is not None:
        algos.append(bt.algos.WeighInvVol(
            **(inv_vol if isinstance(inv_vol, dict) else {})
        ))

    elif erc is not None:
        algos.append(bt.algos.WeighERC(
            **(erc if isinstance(erc, dict) else {})
        ))

    elif mean_var is not None:
        algos.append(bt.algos.WeighMeanVar(
            **(mean_var if isinstance(mean_var, dict) else {})
        ))

    elif target_vol is not None:
        algos.append(bt.algos.TargetVol(
            **(target_vol if isinstance(target_vol, dict) else {})
        ))

    else:
        assert False, weight_kwargs

    if weight_limit is not None:
        algos.append(bt.algos.LimitWeights(weight_limit))

    return algos

# ---------------------------------------------------------------

def build_fixed(
    acc,
    name,
    tickers,
    gross=1.,
    after_days=None,
    frequency=bt.algos.RunDaily(),
    **weight_kwargs,
):

    algos = []

    if "mean_var" in weight_kwargs:
        assert after_days is not None

    if "erc" in weight_kwargs:
        assert after_days is not None

    if after_days is not None:
        algos.append(bt.algos.RunAfterDays(after_days))

    algos.append(frequency)
    algos.append(bt.algos.SelectThese(tickers))

    algos.extend(weight_algos(
        **weight_kwargs
    ))

    algos.append(bt.algos.ScaleWeights(gross))
    algos.append(bt.algos.Rebalance())

    return acc, bt.Strategy(name, algos)

# ---------------------------------------------------------------

def build_rolling(
    acc,
    name,
    universe_df_name,
    gross=1.,
    #
    frequency=bt.algos.RunDaily(),
    after_days=None,
    #
    n=None,
    n_stat_name=None,
    reverse=False,
    #
    **weight_kwargs,
): 
    # assert ls is not None, ls
    # assert n is not None, n

    algos = []

    if after_days is not None:
        algos.append(bt.algos.RunAfterDays(after_days))

    algos.append(frequency)
    algos.append(bt.algos.SelectWhere(universe_df_name))

    if n is not None:
        assert n_stat_name is not None
        algos.append(bt.algos.SetStat(n_stat_name))
        algos.append(bt.algos.SelectN(
            # assumes temp['stat'] available
            # default to top n (if not reverse == True)
            n,
            filter_selected=True,
            sort_descending=not reverse,
        ))

    algos.extend(weight_algos(
        **weight_kwargs
    ))
    
    algos.append(bt.algos.ScaleWeights(gross))
    algos.append(bt.algos.Rebalance())

    # stack is to merge  a bunch of algos into a single node
    # non sequential
    # eg. for a branch

    return acc, bt.Strategy(name, algos)

# ---------------------------------------------------------------

def strategy_basket(
    acc,
    name,
    strats,
    gross=1.,
    frequency=None,
    **weight_kwargs
):
    return build_fixed(
        acc,
        name,
        [
            #
            strat.name for strat in strats
        ],
        gross=gross,
        frequency=frequency,
        **weight_kwargs,
        #
    )

# ---------------------------------------------------------------

    # # fkws = dict(
    # #     fsignals=fsignals,
    # #     fweights=fweights,
    # # )
    # # assert len(v for v in fkws.values() if v) == 1, fkws

    # indices = universe.get("indices", xt.iTuple())
    # sectors = universe.get("sectors", xt.iTuple())
    # tickers = universe.get("ticker", xt.iTuple())

    # if not universe.get("rolling"):
    #     assert len(indices) == 0, universe
    #     assert len(sectors) == 0, universe

    # universe_dfs, _, _ = bt.algos.universe.int.rolling_indices(
    #     date_start,
    #     date_end,
    #     indices=indices,
    #     sectors=sectors,
    # )

def resample_universe(
    df, resampling, lookback=None
):
    df = df.fillna(0)

    if lookback is not None:
        unit = lookback[-1]
        n = int(lookback[:-1]) - 1
        assert unit == resampling, dict(
            unit=unit,
            resampling=resampling,
        )
    else:
        n = 0

    df_periods = df.resample(resampling)

    index_l = xt.iTuple(df_periods.first().index.values)
    index_r = xt.iTuple(df_periods.last().index.values)

    res = df.copy()

    # print("Resampling:", resampling, lookback)

    for l, start, r in zip(
        index_l,
        tuple([None for _ in range(n)]) + index_l,
        index_r,
    ):
        df_slice = df.loc[
            (df.index >= (
                l if start is None else start
            )) & (df.index <= r)
        ]

        res_loc = (res.index >= l) & (res.index <= r)
        in_period = df_slice.all(axis=0)

        res.loc[res_loc] = utils.shapes.expand_dims(
            in_period.values, 0, len(df_slice.index)
        )

        for ticker in df.columns:
            res.loc[res_loc, ticker] = in_period[ticker]

    # print("Resampled:", resampling, lookback)

    res[res.columns] = res[res.columns].astype("bool")
    return res

def index_union(membership_dfs):

    res = {}

    for _, df in membership_dfs.items():
        for ticker in df.columns:
            if ticker not in res:
                res[ticker] = df[ticker]
            else:
                res[ticker] = res[ticker] | df[ticker]

    return pandas.DataFrame(res)

def build(
    acc,
    universe_dfs,
    name=None,
    tickers=xt.iTuple(),
    rolling_universe=True,
    rolling_lookback=None,
    **kws,
):
    if "universes" not in acc:
        acc["universes"] = []
    
    assert name is not None
    assert universe_dfs is not None

    universe_df = index_union(universe_dfs)
    for ticker in tickers:
        universe_df[ticker] = pandas.Series(
            index=universe_df.index,
            data=[True for _ in universe_df.index],
        )

    if rolling_universe:
        universe_df_name = "{}_universe".format(name)

        universe_df = pandas.DataFrame(
            universe_df.values,
            columns=universe_df.columns,
            index=utils.dates.date_index(universe_df.index.values)
        )

        if isinstance(rolling_universe, str):
            universe_df = resample_universe(
                universe_df,
                rolling_universe,
                lookback=rolling_lookback,
            )

        acc[universe_df_name] = universe_df
        acc["universes"].append(universe_df_name)

        return build_rolling(
            acc,
            name,
            universe_df_name=universe_df_name,
            **kws,
        )
    else:
        universe_df_name = "{}_universe".format(name)

        df_index = utils.dates.date_index(
            utils.dates.between(
                min(universe_df.index),
                max(universe_df.index),
            )
        )
        df_universe = pandas.DataFrame({
            ticker: [True for _ in df_index]
            for ticker in tickers
        })
        df_universe.index = df_index

        acc[universe_df_name] = df_universe
        acc["universes"].append(universe_df_name)

        return build_fixed(
            acc,
            name,
            tickers=tickers,
            **kws,
        )

# ---------------------------------------------------------------

def to_prices(df_returns):
    vs = numpy.nan_to_num(df_returns.values)
    return pandas.DataFrame(
        100 * numpy.cumprod(1 + vs, axis = 0),
        columns=df_returns.columns,
        index=utils.dates.date_index(df_returns.index),
    )

def clean_universe(df_universe, df_prices):
    return df_universe.loc[
        df_universe.index.isin(df_prices.index)
    ]

def clean_universes(acc, df_prices):
    return {
        **acc,
        **{
            universe: clean_universe(
                acc[universe], df_prices
            )
            for universe in acc["universes"]
        }
    }

def run(
    strategy,
    df_returns,
    acc,
):
    df_prices = to_prices(df_returns)
    acc = clean_universes(
        acc,
        df_prices
    )
    return bt.run(bt.Backtest(
        strategy,
        df_prices,
        integer_positions=False,
        additional_data=acc,
        progress_bar=True,
    ))

# ---------------------------------------------------------------
