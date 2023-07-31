
import pandas
import numpy

# ---------------------------------------------------------------

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
