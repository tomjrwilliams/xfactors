
import numpy

tolerances=dict(
    rtol=1e-3,
    atol=1e-2,
)

def assert_is_close(v1, v2, b, results=None, n_max=0, **tols):
    if results is None:
        results = dict(v1=v1, v2=v2)
    tols = {**tolerances, **tols}
    diff = numpy.subtract(v1, v2)

    n_diff = ((
        numpy.isclose(v1, v2, **tols)
    ) != b).sum()

    assert n_diff <= n_max, dict(
        **results,
        diff=diff,
        n_diff=n_diff,
    )
