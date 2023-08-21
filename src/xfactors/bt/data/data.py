from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from . import curves as _curves
from . import gics as _gics
from . import indices as _indices

from ... import utils

# ---------------------------------------------------------------

def set_index(df, index_col = "Unnamed: 0"):
    df.index = [
        datetime.date.fromisoformat(v)
        for v in df[index_col].values
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
    """
    >>> df = index_data("SX7E Index")
    >>> round(numpy.mean([
    ...     sum(numpy.isnan(df[c])) / len(df[c])
    ...     for c in df.columns
    ... ]), 3)
    0.0
    """
    fp = index_fp(index, dp=dp)
    return set_index(pandas.read_csv(fp, compression = {
        'method': 'gzip', 'compresslevel': 1, 'mtime': 1
    }))

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
    """
    >>> df = returns_data("SX7E Index")
    >>> round(numpy.mean([
    ...     sum(numpy.isnan(df[c])) / len(df[c])
    ...     for c in df.columns
    ... ]), 3)
    0.53
    """
    fp = returns_fp(index, dp=dp)
    return set_index(pandas.read_csv(fp, compression = {
        'method': 'gzip', 'compresslevel': 1, 'mtime': 1
    }))

# ---------------------------------------------------------------

def curve_fp(
    curve_name,
    dp="./__local__/csvs"
):
    assert " " not in curve_name
    return "{}/curves/{}.csv.zip".format(dp, curve_name)

def sort_tenors(tenors):
    day_map = {
        "M": 30,
        "Y": 365,
    }
    return xt.iTuple(tenors).sort(
        lambda t: int(t[:-1]) * day_map[t[-1]]
    )

def curve_tenor_data(df, tenor):
    return pandas.Series(
        df["yield"].values,
        index=utils.dates.date_index([
            datetime.date.fromisoformat(v)
            for v in df["date"]
        ]),
        name=tenor,
    )

def curve_data(
    curve_name,
    dp="./__local__/csvs"
):
    """
    >>> df = curve_data("YCSW0023")
    >>> {
    ...     c: round(sum(numpy.isnan(df[c])) / len(df[c]), 3)
    ...     for c in df.columns
    ... }
    {'2M': 0.998, '3M': 0.002, '6M': 0.008, '9M': 0.012, '12M': 0.014, '15M': 0.013, '18M': 0.021, '21M': 0.927, '2Y': 0.0, '3Y': 0.0, '4Y': 0.0, '5Y': 0.0, '6Y': 0.0, '7Y': 0.0, '8Y': 0.0, '9Y': 0.0, '10Y': 0.0, '11Y': 0.0, '12Y': 0.0, '15Y': 0.0, '20Y': 0.0, '25Y': 0.0, '30Y': 0.0, '40Y': 0.0, '50Y': 0.0}
    """
    df = pandas.read_csv(
        curve_fp(curve_name, dp=dp), compression="zip",
    )
    return pandas.concat([
        curve_tenor_data(df[df["tenor"] == tenor], tenor)
        for tenor in sort_tenors(df["tenor"].unique())
    ], axis=1)

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
    assert len(indices) or len(sectors)

    res = {}

    for index in indices.sort():
        res[index.split(" ")[0]] = index_data(index, dp=dp)

    for sector in sectors.sort():
        res[_gics.SECTOR_MAP_SHORT.get(
            sector, sector
        )] = index_data(sector, dp=dp)

    if len(indices) + len(sectors) == 1:
        return res[indices.extend(sectors).last()]

    return res

# ---------------------------------------------------------------

def returns_df(
    indices=xt.iTuple([]),
    sectors=xt.iTuple([]),
    dp="./__local__/csvs"
):
    """
    >>> df = returns_df(indices=_indices.EU_MAJOR, sectors=_gics.SECTORS)
    >>> round(numpy.mean([
    ...     sum(numpy.isnan(df[c])) / len(df[c])
    ...     for c in df.columns
    ... ]), 3)
    0.478
    """
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

def curve_dfs(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
):
    """
    >>> dfs = curve_dfs(curves=_curves.CORP_USD)
    >>> xt.iTuple.from_keys(dfs)
    iTuple('USD-AA', 'USD-A', 'USD-BBB', 'USD-BB', 'USD-B')
    """

    assert len(curves), curves

    dfs = curves.map(functools.partial(
        curve_data, dp=dp
    )).zip(curves).mapstar(
        lambda df, curve: df.rename(columns={
            tenor: "{}-{}".format(
                _curves.FULL_MAP[curve],
                tenor
            )
            for tenor in df.columns
        })
    )

    return {
        _curves.FULL_MAP[curve]: df
        for df, curve in dfs.zip(curves)
    }

def curve_df_tall(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
):
    return

# ---------------------------------------------------------------
