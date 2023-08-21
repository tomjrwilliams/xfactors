from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from . import curves
from . import gics
from . import indices

# ---------------------------------------------------------------

def set_index(df):
    if "Unnamed: 0" in df.columns:
        df.index = [
            datetime.date.fromisoformat(v)
            for v in df["Unnamed: 0"].values
        ]
    return df[[c for c in df.columns if "Unnamed" not in c]]

# ---------------------------------------------------------------

def index_fp(
    index,
    dp="./__local__/csvs"
):
    name = index.replace(" ", "_").replace("\\", "")
    return "{}/indices/{}.csv".format(dp, name)

def index_data(
    index,
    dp="./__local__/csvs",
):
    fp = index_fp(index, dp=dp)
    return pandas.read_csv(fp, compression = {
        'method': 'gzip', 'compresslevel': 1, 'mtime': 1
    })

def returns_fp(
    index,
    dp="./__local__/csvs"
):
    name = index.replace(" ", "_").replace("\\", "")
    return "{}/returns/{}.csv".format(dp, name)

def returns_data(
    index,
    dp="./__local__/csvs",
):
    fp = returns_fp(index, dp=dp)
    return pandas.read_csv(fp, compression = {
        'method': 'gzip', 'compresslevel': 1, 'mtime': 1
    })

# ---------------------------------------------------------------

def curve_fp(
    curve_name,
    dp="./__local__/csvs"
):
    assert " " not in curve_name
    return "{}/curves/{}.csv.zip".format(dp, curve_name)

def curve_data(
    curve_name,
    dp="./__local__/csvs"
):
    return pandas.read_csv(
        curve_fp(curve_name, dp=dp), compression="zip",
    )

# ---------------------------------------------------------------

def universe_union(dfs):

    res = {}

    for _, df in dfs.items():
        for ticker in df.columns:
            if ticker not in res:
                res[ticker] = df[ticker]
            else:
                res[ticker] = res[ticker] | df[ticker]

    return pandas.DataFrame(res)

def universe_df(
    indices=xt.iTuple([]),
    sectors=xt.iTuple([]),
    dp="./__local__/csvs"
):
    res = {}

    # TODO: sector renaming here

    for index in indices.sort():
        res[index] = index_data(index, dp=dp)

    for sector in sectors.sort():
        res[sector] = index_data(sector, dp=dp)

    if (len(indices) + len(sectors)) == 1:
        return res[indices.extend(sectors).last()]

    return res

# ---------------------------------------------------------------

def returns_df(
    indices=xt.iTuple([]),
    sectors=xt.iTuple([]),
    dp="./__local__/csvs"
):
    assert len(indices) or len(sectors), dict(
        indices=indices,
        sectors=sectors,
    )

    index_dfs = indices.map(functools.partial(
        returns_data, dp=dp
    ))
    sector_dfs = sectors.map(functools.partial(
        returns_data, dp=dp
    ))

    dfs = index_dfs.extend(sector_dfs)

    if len(dfs) == 1:
        return dfs[0]

    res = {}
    
    for df in dfs:
        for ticker in df.columns:
            if ticker not in res and "Unnamed" not in ticker:
                res[ticker] = df[ticker]

    return pandas.DataFrame(res)

# ---------------------------------------------------------------

def curve_df(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
):
    assert len(curves), curves

    dfs = curves.map(functools.partial(
        curve_data, dp=dp
    )).zip(curves).mapstar(
        lambda df, curve: df.rename(columns={
            "{}-{}".format(
                curves.FULL_MAP[curve],
                tenor
            )
            for tenor in df.columns
        })
    )

    return {
        curve.FULL_MAP[curve]: df
        for df, curve in dfs.zip(curves)
    }

def curve_df_tall(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
):
    return

# ---------------------------------------------------------------
