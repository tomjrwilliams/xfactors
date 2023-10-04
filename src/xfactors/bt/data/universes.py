from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from . import gics as _gics
from . import indices as _indices

from ... import utils
from ... import visuals

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

def universe_mapping(
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

def set_index(df, index_col = "Unnamed: 0"):
    df.index = utils.dates.date_index([
        datetime.date.fromisoformat(v)
        for v in df[index_col].values
    ])
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

# ---------------------------------------------------------------
