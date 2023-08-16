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


def cache_df_csv(f_name, f_data, dir, categorical_map=False):

    compression = {'method': 'gzip', 'compresslevel': 1, 'mtime': 1}

    if dir[0] == ".":
        dir = os.environ["MODULE"] + dir[1:]

    def res(date_start, date_end, **kwargs):
        key = dict(
            date_start=date_start,
            date_end=date_end,
            **kwargs
        )
        name = f_name(**key)

        dp = pathlib.Path(dir)
        if not dp.exists():
            dp.mkdir(exist_ok=True, parents=True)

        fp_json = dp / "{}.json".format(name)
        fp_csv = dp / "{}.csv".format(name)

        if fp_json.exists():
            
            with fp_json.open('r') as f:
                blob = json.load(f)

            blob_key = blob["key"]

            blob_start = datetime.datetime.fromisoformat(
                blob_key["date_start"]
            ).date()
            blob_end = datetime.datetime.fromisoformat(
                blob_key["date_end"]
            ).date()
 
            if blob_start <= date_start and blob_end >= date_end:
                data = pandas.read_csv(fp_csv, compression = compression)
                return data

            date_start=min([date_start, blob_start])
            date_end=min([date_end, blob_end])

            key={
                **key,
                **dict(
                    date_start=date_start,
                    date_end=date_end,
                )
            }

        blob = dict(key=key)
        data = f_data(**key)

        data.to_csv(fp_csv, compression = compression)

        with fp_json.open('w+') as f:
            json.dump({
                **blob,
                **dict(key={
                    **key,
                    **dict(
                        date_start=date_start.isoformat(),
                        date_end=date_end.isoformat(),
                    )
                }),
            }, f)

        return data
    
    return res

def returns_df_index_name(date_start, date_end, index):
    return index.replace(" ", "_").replace("\\", "")

returns_df_index = cache_df_csv(
    returns_df_index_name,
    returns_df_index_data,
    "./__local__/csvs/returns"
)

def returns_df_index_data(
    date_start,
    date_end,
    index,
):
    universe = hcbt.algos.universe.int.rolling_index(
        date_start,
        date_end,
        index=index,
    )
    universe = universe[
        [
            col for col in universe if "Unnamed" not in col
        ]
    ]

    returns = returns_df(
        date_start,
        date_end,
        tickers=xt.iTuple.from_columns(universe)
    )
    return returns

def returns_df_index_name(date_start, date_end, index):
    return index.replace(" ", "_").replace("\\", "")

returns_df_index = cache_df_csv(
    returns_df_index_name,
    returns_df_index_data,
    "./__local__/csvs/returns"
)

def set_index(df):
    if "Unnamed: 0" in df.columns:
        df.index = [
            datetime.date.fromisoformat(v)
            for v in df["Unnamed: 0"].values
        ]
    return df[[c for c in df.columns if "Unnamed" not in c]]

def returns_df(
    date_start,
    date_end,
    indices=xt.iTuple([]),
    sectors=xt.iTuple([]),
):
    assert len(indices) or len(sectors)
    len_total = len(indices) + len(sectors)

    index_dfs = indices.map(
        lambda i: dict(index=i)
    ).map(lambda i: set_index(
        returns_df_index(date_start, date_end, **i)
    ))

    sector_dfs = sectors.map(
        lambda i: dict(index=i)
    ).map(lambda i: set_index(
        returns_df_index(date_start, date_end, **i)
    ))

    if len_total == 1 and len(indices) == 1:
        return index_dfs[0]
    
    if len_total == 1 and len(sectors) == 1:
        return sector_dfs[0]

    res = {}
    
    for df in list(index_dfs) + list(sector_dfs):
        for ticker in df.columns:
            if ticker not in res and "Unnamed" not in ticker:
                res[ticker] = df[ticker]

    return pandas.DataFrame(res)