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
from ... import visuals

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

def sort_tenors(tenors, with_indices = False, reverse=False):
    day_map = {
        "M": 30,
        "Y": 365,
    }
    f_sort = lambda t: int(t[:-1]) * day_map[t[-1]]
    if with_indices:
        return xt.iTuple(tenors).sort_with_indices(
            f_sort,
            reverse=reverse
        )
    return xt.iTuple(tenors).sort(
        f_sort,
        reverse=reverse
    )

def enumerate_tenors(tenors, i0 = 0, reverse=False):
    """
    >>> enumerate_tenors(["3Y", "3M", "1Y", "1M"])
    iTuple('03-3Y', '01-3M', '02-1Y', '00-1M')
    """
    return sort_tenors(
        tenors, with_indices=True, reverse=reverse
    ).enumerate().mapstar(
        lambda i, it: (it[0], "{}-{}".format(
            visuals.formatting.left_pad(
                str(i + i0), l=2, pad="0"
            ),
            it[1]
        ))
    ).sortstar(lambda i, t: i).mapstar(lambda i, t: t)
    
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

def sort_curves(curves, with_indices = False, reverse=False):
    f_sort = lambda pref, suff: pref + {
        "G": "0",
        "S": "1",
    }.get(suff, "2{}{}".format(
        suff[0],
        3 - len(suff)
    ))
    f_split_sort = lambda c: f_sort(*c.split("-"))
    if with_indices:
        return xt.iTuple(curves).sort_with_indices(
            f_split_sort,
            reverse=reverse
        )
    return xt.iTuple(curves).sort(
        f_split_sort,
        reverse=reverse
    )

def enumerate_curves(curves, i0 = 0, reverse=False):
    """
    >>> enumerate_curves(["USD-B", "EUR-AA", "USD-G", "USD-AA", "EUR-S"])
    iTuple('04-USD-B', '01-EUR-AA', '02-USD-G', '03-USD-AA', '00-EUR-S')
    """
    return sort_curves(
        curves, with_indices=True, reverse=reverse
    ).enumerate().mapstar(
        lambda i, ic: (ic[0], "{}-{}".format(
            visuals.formatting.left_pad(
                str(i + i0), l=2, pad="0"
            ),
            ic[1]
        ))
    ).sortstar(lambda i, c: i).mapstar(lambda i, c: c)

# reverse = descending
def enumerate_strip_curve(df, curve, reverse=False):
    col_tenors = {
        col: col.replace("{}-".format(curve), "")
        for col in df.columns if curve in col
    }
    df = df.rename(columns={
        col: enum_tenor
        for enum_tenor, (col, tenor) in enumerate_tenors(
            col_tenors.values(), reverse=reverse
        ).zip(col_tenors.items())
    })
    df = df[list(sorted(df.columns))]
    return df

def enumerate_strip_tenor(df, tenor, reverse=False):
    col_curves = {
        col: col.replace("-{}".format(tenor), "")
        for col in df.columns if tenor in col
    }
    df = df.rename(columns={
        col: enum_curve
        for enum_curve, (col, curve) in enumerate_curves(
            col_curves.values(), reverse=reverse
        ).zip(col_curves.items())
    })
    df = df[list(sorted(df.columns))]
    return df

def curves_by_tenor(
    dfs, tenor = None
):
    if isinstance(dfs, dict):
        dfs = xt.iTuple.from_values(dfs)

    tenors = (
        dfs.map(xt.iTuple.from_columns)
        .flatten()
        .map(lambda c: c.split("-")[-1])
        .pipe(sort_tenors)
    )
    if tenor is None:
        return {
            tenor: pandas.concat([
                df[[col for col in df.columns if tenor in col]]
                for df in dfs
            ], axis=1)
            for tenor in tenors
        }
    return pandas.concat([
        df[[col for col in df.columns if tenor in col]]
        for df in dfs
    ], axis=1)

def curve_dfs(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
    merge=False,
    by_tenor=False,
):
    """
    >>> dfs = curve_dfs(curves=_curves.CORP_USD)
    >>> xt.iTuple.from_keys(dfs)
    iTuple('USD-AA', 'USD-A', 'USD-BBB', 'USD-BB', 'USD-B')
    >>> len(curve_dfs(curves=_curves.CORP_USD, merge=True).columns)
    79
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
    if merge:
        return pandas.concat(dfs, axis=1)

    if by_tenor:
        return curves_by_tenor(dfs)

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
