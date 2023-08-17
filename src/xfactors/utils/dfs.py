
import pandas
import numpy


import xtuples as xt

# ---------------------------------------------------------------

def shift(df, shift, fill = numpy.NaN):
    if shift is None:
        return df
    units = shift[-1]
    periods = int(shift[:-1])
    assert units == "D", shift # else we gotta get fancy
    return pandas.DataFrame(
        numpy.concatenate([
            numpy.ones((periods, len(df.columns,))) * fill,
            df.values[periods:]
        ], axis = 0),
        index=df.index,
        columns=df.columns,
    )

def merge_indices(dfs):
    index = dfs[0].index
    for df in dfs[1:]:
        index = index.union(df.index)
    return index

def index_date_filter(df, date_start=None, date_end=None):
    if date_start is not None:
        df = df.loc[df.index >= date_start]
    if date_end is not None:
        df = df.loc[df.index <= date_end]
    return df

def df_min_max(
    df,
    with_cols=None,
    excl_cols=None,
    symmetrical=False,
):
    ks = list(df.columns)
    if with_cols:
        ks = [k for k in ks if k in with_cols]
    if excl_cols:
        ks = [k for k in ks if k not in excl_cols]
    df_vs = df[ks].values
    v_min = numpy.min(df_vs)
    v_max = numpy.max(df_vs)
    if v_min < 0 and v_max < 0:
        pass
    elif v_min > 0 and v_max > 0:
        pass
    elif not symmetrical:
        pass
    else:
        v_lim = max([abs(v_min), v_max])
        v_min = -1 * v_lim
        v_max = v_lim
    return v_min, v_max

def melt_with_index(
    df, 
    index_as="index",
    variable_as="variable",
    value_as="value",
    columns = None
):
    if columns is None:
        columns = df.columns
    if len(df.index) < len(df.columns):
        res = pandas.concat([
            df[columns].loc[[v]].melt().assign(
                **{index_as: [v for _ in df.columns]}
            ) for v in df.index
        ])
        return res.rename({
            "variable": variable_as,
            "value": value_as,
        }, axis=1, inplace=False)
    return pandas.concat([
        pandas.DataFrame({
            variable_as: [col for _ in df.index],
            value_as: df[col].values,
            index_as: df.index.values
        })
        for col in columns
    ])

# ---------------------------------------------------------------

def rolling_windows(
    df, lookback
):
    unit = lookback[-1]
    n = int(lookback[:-1]) - 1

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

    for l, start, r in zip(
        index_l,
        tuple([None for _ in range(n)]) + index_l,
        index_r,
    ):
        l = l if start is None else start
        yield l, r, df.loc[
            (df.index >= (
                l if start is None else start
            )) & (df.index <= r)
        ]
        
# ---------------------------------------------------------------
