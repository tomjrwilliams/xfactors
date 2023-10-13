
from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from . import gics as _gics
from . import indices as _indices
from . import universes

from ... import utils
from ... import visuals

set_index = universes.set_index

# ---------------------------------------------------------------

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
