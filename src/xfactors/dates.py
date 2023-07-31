
import functools
import datetime

import xtuples as xt

def y(y, m = 1):
    return datetime.date(y, m, 1)

def iter_dates(d, n = None, it = 1):
    if n is None:
        def f():
            step = datetime.timedelta(days=it)
            yield d
            while True:
                d += step
                yield d
    def f():
        return (
            xtuples.iTuple.range(n)
            .map(lambda i: d + datetime.timedelta(days=i * it))
        )
    return f()

starting = functools.partial(iter_dates, it = 1)
ending = functools.partial(iter_dates, it = -1)

def between(d1, d2):
    if d2 > d1:
        return starting(d1, (d2 - d1).days)
    return starting(d2, (d1 - d2).days).reverse()

import pandas

def dated_series(dd):
    return pandas.Series(
        data=list(dd.values()),
        index=pandas.DatetimeIndex(data=list(dd.keys())),
    )