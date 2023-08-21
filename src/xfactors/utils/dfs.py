
import pandas
import numpy


import xtuples as xt

from . import dates
from . import shapes

# ---------------------------------------------------------------

def shift(df, shift, fill = numpy.NaN, calendar = None):
    if shift is None:
        return df
    units = shift[-1]
    periods = int(shift[:-1])
    if calendar is None:
        assert units == "D", shift # else we gotta get fancy
        return pandas.DataFrame(
            numpy.concatenate([
                numpy.ones((periods, len(df.columns,))) * fill,
                df.values[periods:]
            ], axis = 0),
            index=df.index,
            columns=df.columns,
        )
    elif calendar=="FULL":
        return df.shift(periods=periods, freq=units)
    else:
        assert False, calendar

def merge_indices(dfs):
    index = dfs[0].index
    for df in dfs[1:]:
        index = index.union(df.index)
    return index

def index_date_filter(df, date_start=None, date_end=None):
    if date_start is not None:
        df = df.loc[df.index >= pandas.to_datetime(date_start)]
    if date_end is not None:
        df = df.loc[df.index <= pandas.to_datetime(date_end)]
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

    df.index = dates.date_index(df.index.values)

    index_l = xt.iTuple(
        df.resample("{}{}".format(n, unit), label="left")
        .first()
        .index
        .values
    )
    index_r = xt.iTuple(
        df.resample("{}{}".format(n, unit), label="right")
        .last()
        .index
        .values
    )

    for l, start, r in zip(
        index_l,
        tuple([None for _ in range(n)]) + index_l,
        index_r,
    ):
        l = min(index_l) if start is None else start
        yield l, r, df.loc[
            (df.index >= l) & (df.index <= r)
        ]

def rolling_apply(
    f,
    df,
    lookback,
    df_mask = None,
    na_threshold_row=0.,
    na_threshold_col=0.,
):
    if df_mask is not None:
        df_columns = xt.iTuple(df.columns)
        df_mask_columns = xt.iTuple(df_mask.columns)
        assert df_columns == df_mask_columns, dict(
            df=df_columns,
            df_mask=df_mask_columns,
        )
    df.index = dates.date_index(df.index.values)
    df_res = pandas.DataFrame(
        numpy.zeros_like(df.values),
        columns=df.columns,
        index=df.index,
    )
    if df_mask is None:
        df_mask = pandas.DataFrame(
            numpy.ones_like(df_res.values),
            columns=df.columns,
            index=df.index,
        )
    for l, r, df_slice in rolling_windows(
        df_mask, lookback
    ):
        nd_universe = df_slice.values[0]
        i_cols = numpy.nonzero(nd_universe)

        loc = (df.index >= l) & (df.index <= r)
        df_slice = df.loc[loc]

        res = numpy.ones(df_slice.shape) * numpy.NaN
        
        nd_slice = df_slice.values[:, i_cols][:, 0, :]
        
        na_row_counts = numpy.isnan(nd_slice).sum(axis=1)
        not_na_rows = numpy.nonzero(
            (na_row_counts / nd_slice.shape[1]) <= na_threshold_row
        )

        nd_slice = nd_slice[not_na_rows]

        is_na_col = numpy.isnan(nd_slice).sum(axis=0)
        not_na_col = (
            is_na_col / nd_slice.shape[0]
        ) <= na_threshold_col

        nd_universe[i_cols] = nd_universe[i_cols] * not_na_col
        i_cols = numpy.nonzero(nd_universe)

        nd_slice = df_slice.values[not_na_rows]
        nd_slice = nd_slice[:, i_cols][:, 0, :]

        res_r = numpy.zeros(len(df_slice.columns))
        
        if not nd_slice.shape[0] or not nd_slice.shape[1]:
            df_res.loc[loc] = res
        
        else:
            res_r[i_cols] = f(nd_slice)
            res[not_na_rows] = numpy.array(
                shapes.expand_dims(res_r, 0, len(not_na_rows))
            )
            
            df_res.loc[loc] = res
    
    df_res.index = dates.date_index(df_res.index.values)

    return df_res


# ---------------------------------------------------------------
