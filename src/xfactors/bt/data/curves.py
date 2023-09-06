
from __future__ import annotations

import functools

import datetime

import numpy
import pandas

import pathlib

import xtuples as xt

from ... import utils
from ... import visuals

# ---------------------------------------------------------------

FULL_MAP = {
    # USD
    "YCSW0023 Index": "USD-S",
    "YCGT0025 Index": "USD-G",
    "YCGT0169 Index": "USD-I",
    # # EUR
    "YCSW0045 Index": "EUR-S",
    "YCSW0092 Index": "EUR-S",
    # USD CORP IG CUrve
    "BVSC0076 Index": "USD-IG",
    # AA
    "BVSC0073 Index": "USD-AA",
    # A
    "BVSC0074 Index": "USD-A",
    # bbb
    "BVSC0075 Index": "USD-BBB",
    # bb
    "BVSC0193 Index": "USD-BB",
    # b
    "BVSC0195 Index": "USD-B",
    # EUR
    # aa
    "BVSC0165 Index": "EUR-AA",
    # a
    "BVSC0077 Index": "EUR-A",
    # bbb
    "BVSC0166 Index": "EUR-BBB",
    # JP
    # aa
    "BVSC0153 Index": "JPY-AA",
    # a
    "BVSC0154 Index": "JPY-A",
    # # DE
    "YCGT0016 Index": "EUR-DE",
    # # FR
    "YCGT0014 Index": "EUR-FR",
    # # JP
    "YCGT0018 Index": "JPY-G",
    "YCSW0097 Index": "JPY-S",
    "YCGT0385 Index": "JPY-I",
    # # UK
    "YCGT0022 Index": "GBP-G",
    "YCSW0022 Index": "GBP-S",
    # # AU
    "YCGT0001 Index": "AUD-G",
    "YCSW0001 Index": "AUD-S",
    "YCGT0204 Index": "AUD-I",
    # # IT
    "YCGT0040 Index": "EUR-IT",
    # "YCGT0331 Index": "", inflation?
    # # CA
    "YCGT0007 Index": "CAD-G",
    # # CN
    "YCGT0299 Index": "CHN-G",
    # # SP
    "YCGT0061 Index": "EUR-ES",
    # # SW
    "YCGT0082 Index": "CHF-G",
    # # SE
    "YCGT0021 Index": "SEK-G",
    # NZ
    "YCGT0049 Index": "NZD-G",
}
FULL_MAP = {
    k.split(" ")[0]: v for k, v in FULL_MAP.items()
}
FULL = xt.iTuple.from_keys(FULL_MAP)

def is_corp(ccy, suffix, *, with_ccy = None):
    return any([
        rating in suffix for rating in ["A", "B"]
    ]) and (
        True if not with_ccy else ccy == with_ccy
    )

CORP_USD_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="USD")
}
CORP_USD = xt.iTuple.from_keys(CORP_USD_MAP)
CORP_USD_NAMES = xt.iTuple.from_values(CORP_USD_MAP)

CORP_EUR_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="EUR")
}
CORP_EUR = xt.iTuple.from_keys(CORP_EUR_MAP)
CORP_EUR_NAMES = xt.iTuple.from_values(CORP_EUR_MAP)

CORP_JPY_MAP = {
    k: v for k, v in FULL_MAP.items()
    if is_corp(*v.split("-"), with_ccy="JPY")
}
CORP_JPY = xt.iTuple.from_keys(CORP_JPY_MAP)
CORP_JPY_NAMES = xt.iTuple.from_values(CORP_JPY_MAP)

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
                str(i + i0), l=2, pad="0", sign=False
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
                str(i + i0), l=2, pad="0", sign=False
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
                FULL_MAP[curve],
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
        FULL_MAP[curve]: df
        for df, curve in dfs.zip(curves)
    }

def curve_df_tall(
    curves=xt.iTuple(),
    dp="./__local__/csvs",
):
    return

# ---------------------------------------------------------------

def curve_chart(dfs_curves, d_start, d_end, curve):
    df = dfs_curves[curve]
    df = utils.dfs.index_date_filter(
        df, date_start=d_start, date_end=d_end
    )
    df = enumerate_strip_curve(df, curve, reverse=True)
    return visuals.graphs.df_line_chart(
        utils.dfs.melt_with_index(
            df, index_as="date", variable_as="tenor"
        ),
        x="date",
        y="value",
        color="tenor",
        discrete_color_scale="Blues",
    )


def tenor_chart(dfs_curves, d_start, d_end, tenor, curves):
    df = curves_by_tenor({
        curve: df for curve, df in dfs_curves.items() 
        if curve in curves
    }, tenor = tenor)
    df = utils.dfs.index_date_filter(
        df, date_start=d_start, date_end=d_end
    )
    df = enumerate_strip_tenor(df, tenor, reverse=True)
    return visuals.graphs.df_line_chart(
        utils.dfs.melt_with_index(df, index_as="date", variable_as="curve"),
        x="date",
        y="value",
        color="curve",
        discrete_color_scale="Blues",
    )
    
# tenor_chart(
#     xf.utils.dates.y(2005),
#     xf.utils.dates.y(2023),
#     curves=xt.iTuple.from_keys(dfs_curves).filter(lambda s: "USD" in s),
#     tenor="3M",
# )

def curve_cov(
    dfs_curves, curve, d_start, d_end, v_min=None, v_max=None
):
    df = dfs_curves[curve]
    df = utils.dfs.index_date_filter(
        df, date_start=d_start, date_end=d_end
    )
    return visuals.rendering.render_df_color_range(
        df.cov(), v_min=v_min, v_max=v_max
    )
# curve_cov("USD-S", xf.utils.dates.y(2005), xf.utils.dates.y(2023))

def curve_corr(
    dfs_curves, curve, d_start, d_end, v_min=-1., v_max=1.,
):
    df = dfs_curves[curve]
    df = utils.dfs.index_date_filter(
        df, date_start=d_start, date_end=d_end
    )
    return visuals.rendering.render_df_color_range(
        df.corr(),
        v_min=v_min,
        v_max=v_max,
    )
# curve_corr("USD-S", xf.utils.dates.y(2005), xf.utils.dates.y(2023))

# ---------------------------------------------------------------
