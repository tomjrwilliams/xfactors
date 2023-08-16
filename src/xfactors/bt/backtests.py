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
    acc,
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

        assert weight_df_name in acc, dict(
            weight_df=weight_df_name,
            keys=list(acc.keys()),
        )

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

    return acc, algos

# ---------------------------------------------------------------

def build_fixed(
    acc,
    name,
    tickers,
    gross=1.,
    after_days=None,
    frequency=bt.algos.RunDaily(),
    children=None,
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

    acc, algos_weight = weight_algos(
        acc, **weight_kwargs
    )
    algos.extend(algos_weight)

    algos.append(bt.algos.ScaleWeights(gross))
    algos.append(bt.algos.Rebalance())

    return acc, bt.Strategy(name, algos, children=children)

def combine_fixed(
    acc,
    name,
    strats,
    gross=1.,
    after_days=None,
    frequency=bt.algos.RunDaily(),
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
        after_days=after_days,
        children=strats,
        **weight_kwargs,
        #
    )

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
    top_n=None,
    top_n_df_name=None,
    reverse=False,
    #
    children=None,
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

    if top_n is not None:
        assert top_n_df_name is not None

        assert top_n_df_name in acc, dict(
            n_stat=top_n_df_name,
            keys=list(acc.keys()),
        )

        algos.append(bt.algos.SetStat(top_n_df_name))
        algos.append(bt.algos.SelectN(
            # assumes temp['stat'] available
            # default to top n (if not reverse == True)
            top_n,
            filter_selected=True,
            sort_descending=not reverse,
        ))

    acc, algos_weight = weight_algos(
        acc, **weight_kwargs
    )
    algos.extend(algos_weight)
    
    algos.append(bt.algos.ScaleWeights(gross))
    algos.append(bt.algos.Rebalance())

    # stack is to merge  a bunch of algos into a single node
    # non sequential
    # eg. for a branch

    return acc, bt.Strategy(name, algos, children=children)

def combine_rolling(
    acc,
    name,
    universe_df_name,
    strats,
    gross=1.,
    #
    frequency=bt.algos.RunDaily(),
    after_days=None,
    #
    top_n=None,
    top_n_df_name=None,
    reverse=False,
    #
    **weight_kwargs,
):
    return build_rolling(
        acc,
        name,
        universe_df_name,
        gross=gross,
        frequency=frequency,
        after_days=after_days,
        top_n=top_n,
        top_n_df_name=top_n_df_name,
        reverse=reverse,
        children=strats,
        #
        **weight_kwargs,
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

# for mean_var type optimisation where need a lookback for covar
# so set per window to only universe present in relevant window
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

    index_l = xt.iTuple(
        df.resample("{}{}".format(n, unit), label="left")
        .first()
        .index.values
    )
    index_r = xt.iTuple(
        df.resample("{}{}".format(n, unit), label="right")
        .last()
        .index.values
    )

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

def long_short(
    acc,
    dfs_indices,
    name,
    signal_name="signal",
    long=True,
    short=True,
    strat_kwargs = None,
    combine=True,
    **kwargs,
):
    if strat_kwargs is None:
        strat_kwargs = {}

    assert signal_name in acc
    
    # equal=True,
    # top_n=15,
    # top_n_df_name="signal",

    strats = []
    if long:
        if isinstance(long, bt.Strategy):
            strat_long = long
        else:
            acc, strat_long = build(
                acc,
                dfs_indices,
                name=name + " long",
                rolling_universe=True,
                reverse=False,
                gross=1.,
                equal=strat_kwargs.get("equal", True),
                **{
                    k: v for k, v in strat_kwargs.items()
                    if k != "equal"
                },
            )
        strats.append(strat_long)
    if short:
        if isinstance(short, bt.Strategy):
            strat_short = short
        else:
            acc, strat_short = build(
                acc,
                dfs_indices,
                name=name + " short",
                rolling_universe=True,
                reverse=True, # only has an effect if we use top n
                gross=-1.,
                equal=strat_kwargs.get("equal", True),
                **{
                    k: v for k, v in strat_kwargs.items()
                    if k != "equal"
                },
            )
        strats.append(strat_short)
    assert len(strats) > 0
    if len(strats) == 0:
        return acc, strats[0]
    elif combine:
        acc, strat = combine_fixed(
            acc,
            name,
            strats,
            gross=1.,
            equal=kwargs.get("equal", True),
            # default to true for convenince
            **{
                k: v for k, v in kwargs.items()
                if k != "equal"
            },
        )
        return acc, strat
    return acc, strats

# NOTE: use combine = false
# to build up a list of long short pairs
# to then combine fixed manually
# eg. sector long short baskets
# each separately pass {k: dfs_sectors[k]} for universe

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
