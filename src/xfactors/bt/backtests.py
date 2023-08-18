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
        assert weight_df_name in acc, dict(
            weight_df=weight_df_name,
            keys=list(acc.keys()),
        )
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

# for mean_var type optimisation where need a lookback for covar
# so set per window to only universe present in relevant window
def resample_universe(
    df, lookback
):

    df.index = utils.dates.date_index(df.index.values)

    df = df.fillna(0)
    res = df.copy()

    for l, r, df_slice in utils.dfs.rolling_windows(
        df, lookback
    ):

        res_loc = (res.index >= l) & (res.index <= r)
        in_period = df_slice.all(axis=0)

        res.loc[res_loc] = utils.shapes.expand_dims(
            in_period.values, 0, len(df_slice.index)
        )

        for ticker in df.columns:
            res.loc[res_loc, ticker] = in_period[ticker]

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
    universe_df,
    name=None,
    tickers=xt.iTuple(),
    rolling_universe=True,
    rolling_lookback=None,
    **kws,
):
    if "universes" not in acc:
        acc["universes"] = []
    
    assert name is not None
    assert universe_df is not None

    # universe_df = index_union(universe_df)

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
    universe_df,
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
                universe_df,
                name=name + " long",
                rolling_universe=True,
                reverse=False,
                equal=strat_kwargs.get("equal", True),
                gross=strat_kwargs.get("gross_long", 1.),
                **{
                    k: v for k, v in strat_kwargs.items()
                    if k not in ["equal", "gross"]
                },
            )
        strats.append(strat_long)
    if short:
        if isinstance(short, bt.Strategy):
            strat_short = short
        else:
            acc, strat_short = build(
                acc,
                universe_df,
                name=name + " short",
                rolling_universe=True,
                reverse=True, # only has an effect if we use top n
                equal=strat_kwargs.get("equal", True),
                gross=strat_kwargs.get("gross_short", -1.),
                **{
                    k: v for k, v in strat_kwargs.items()
                    if k not in ["equal", "gross"]
                },
            )
        strats.append(strat_short)
    assert len(strats) > 0
    if len(strats) == 0:
        return acc, strats[0]
    elif combine:
        if "weights" in kwargs:
            weights = {
                ticker: w
                for ticker, w in zip(
                    [strat.name for strat in strats],
                    kwargs["weights"],
                )
            }
        else:
            weights = None
        acc, strat = combine_fixed(
            acc,
            name,
            strats,
            gross=1.,
            equal=kwargs.get("equal", (
                False if "weights" in kwargs else True
            )),
            # default to true for convenince
            weights=weights,
            **{
                k: v for k, v in kwargs.items()
                if k not in ["equal", "weights"]
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
    acc,
    df_returns,
    *strategies
):
    df_prices = to_prices(df_returns)
    acc = clean_universes(
        acc,
        df_prices
    )
    backtests = [
        bt.Backtest(
            strategy,
            df_prices,
            integer_positions=False,
            progress_bar=True,
            additional_data=acc,
        )
        for strategy in strategies
    ]
    return bt.run(*backtests)

# NOTE: multiple backtests for plotting long and short legs separately?

# ---------------------------------------------------------------
