
import datetime

import numpy
import pandas

from .. import utils

# performance assuming weights at date: offset = 0

# actually tradable performance: offset = c. 2 (business) days
# d-1 we get data
# d we make trade
# d+1 we get performance

def shift_weights(df_ws, shift):
    units = shift[-1]
    periods = int(shift[:-1])
    return df_ws.shift(
        periods=periods, freq=units, fill_value=0
    )

def example_dfs():
    """
    >>> df_r, df_w = example_dfs()
    >>> df_r
                  0    1    2
    2020-01-01  0.1 -0.1  0.1
    2020-01-02  0.2 -0.2  0.2
    2020-01-03  0.1 -0.1  0.1
    >>> df_w
                       0         1         2
    2020-01-01  0.333333 -0.333333  0.333333
    2020-01-02 -0.333333  0.333333 -0.333333
    2020-01-03  0.333333 -0.333333  0.333333
    """
    cols = list(range(3))
    index = utils.dates.date_index(utils.dates.starting(
        datetime.date(2020, 1, 1), 3
    ))
    rets = numpy.array([
        [.1, -.1, .1], [.2, -.2, .2], [.1, -.1, .1]
    ])
    ws = numpy.array([
        [1, -1, 1], [-1, 1, -1], [1, -1, 1]
    ]) / 3
    df_r = pandas.DataFrame(rets, index=index, columns=cols)
    df_w = pandas.DataFrame(ws, index=index, columns=cols)
    return df_r, df_w

def df_factor_return(df_rs, df_ws, shift = "1D", cum = False):
    """
    >>> df_factor_return(*example_dfs())
    2020-01-01    0.0
    2020-01-02    0.2
    2020-01-03   -0.1
    2020-01-04    0.0
    dtype: float64
    >>> df_factor_return(*example_dfs(), cum = True)
    2020-01-01    1.00
    2020-01-02    1.20
    2020-01-03    1.08
    2020-01-04    1.08
    dtype: float64
    """
    df_wrs = df_rs.multiply(
        shift_weights(df_ws, shift)
    ).sum(axis=1)
    if cum:
        return (1 + df_wrs).cumprod()
    return df_wrs

def df_factor_trend(
    df_factor, alpha=2 / 30, z = False
):
    """
    >>> df_factor_trend(df_factor_return(*example_dfs()))
    2020-01-01    0.000000
    2020-01-02    0.103448
    2020-01-03    0.030903
    2020-01-04    0.022361
    dtype: float64
    """
    ewm = df_factor.ewm(alpha=alpha)
    if z:
        return ewm.mean() / ewm.std()
    return ewm.mean()

# NOTE: momentum: given factor trend (> threshold), forward return prediction is positive

# NOTE: reversion, given factor trend (> threshold), forward return prediction is negative

def df_factor_betas(
    df_factor, df_rs, alpha = 2 / 30, z = False
):
    """
    >>> df_r, df_w = example_dfs()
    >>> df_factor_betas(df_factor_return(df_r, df_w), df_r)
                       0         1         2
    2020-01-01       NaN       NaN       NaN
    2020-01-02  0.500000 -0.500000  0.500000
    2020-01-03  0.353428 -0.353428  0.353428
    2020-01-04  0.540335 -0.540335  0.540335
    """
    df_factor_ewm = df_factor.ewm(alpha=alpha)
    df_cov = pandas.DataFrame({
        col: df_factor_ewm.cov(df_rs[col])
        for col in df_rs.columns
    })
    df_var = df_factor_ewm.var()
    df_betas = df_cov.divide(df_var, axis = 0) 
    if z:
        return df_betas / df_betas.ewm(alpha=alpha).std()
    return df_betas

# ie. factor spread
# pass in alpha to take rolling mean
# -> rolling spread signal accumulator, assume then converges
# or not
def df_factor_deltas(
    df_betas,
    df_rs,
    alpha = None,
    z = False,
    cum= False,
):
    """
    >>> df_r, df_w = example_dfs()
    >>> df_returns = df_factor_return(df_r, df_w)
    >>> df_betas = df_factor_betas(df_returns, df_r)
    >>> df_factor_deltas(df_betas, df_r)
                       0         1         2
    2020-01-01       NaN       NaN       NaN
    2020-01-02  0.100000 -0.300000  0.100000
    2020-01-03  0.064657 -0.135343  0.064657
    2020-01-04       NaN       NaN       NaN
    """
    df_predictions = df_betas.multiply(df_rs)
    df_deltas = df_rs.subtract(df_predictions)
    if alpha is None:
        return df_deltas
    if cum:
        return (1 + df_deltas).cumprod()
    ewm = df_deltas.ewm(alpha=alpha)
    if z:
        return ewm.mean() / ewm.std()
    return ewm.mean()
