from __future__ import annotations

import os
import pathlib

import functools
import itertools

import typing
import dataclasses

import datetime

import numpy
import pandas

import pathlib

import scipy.stats

import plotly.express

import xtuples as xt

# ---------------------------------------------------------------

def weights_to_membership(weights):
    df = pandas.DataFrame({
        ticker: core.date_dict_to_series(dws).fillna(0)
        for ticker, dws in weights.items()
    })
    df = df != 0
    assert sum(sum(df.values)) > 0, weights[
        list(weights.keys())[0]
    ]
    return df

def rolling_index_data(date_start, date_end, index):
    (
        _, weights
        #
    ) = hcbt.algos.universe.configs.get_rolling_index(
        date_start, date_end, index
    )
    return weights_to_membership(weights)

def rolling_index_f_name(date_start, date_end, index):
    return index.replace(" ", "_").replace("\\", "")

rolling_index = core.cache_df_csv(
    rolling_index_f_name,
    rolling_index_data,
    "./__local__/csvs/indices",
)

def set_index(df):
    if "Unnamed: 0" in df.columns:
        df.index = [
            datetime.date.fromisoformat(v)
            for v in df["Unnamed: 0"].values
        ]
    return df[[c for c in df.columns if "Unnamed" not in c]]

def rolling_indices(
    date_start,
    date_end,
    indices=xt.iTuple([]),
    sectors=xt.iTuple([]),
):
    res = {}

    for index in indices.sort():
        df = set_index(
            rolling_index(date_start, date_end, index=index)
        )
        res[index] = df

    for sector in sectors.sort():
        df = set_index(
            rolling_index(date_start, date_end, index=sector)
        )
        res[sector] = df

    if (len(indices) + len(sectors)) == 1:
        return res[indices.extend(sectors).last()]

    return res
    #, indices, sectors

def index_union(membership_dfs):

    res = {}

    for _, df in membership_dfs.items():
        for ticker in df.columns:
            if ticker not in res:
                res[ticker] = df[ticker]
            else:
                res[ticker] = res[ticker] | df[ticker]

    return pandas.DataFrame(res)


# ---------------------------------------------------------------

# and then the csvs can be dump pulled from the server