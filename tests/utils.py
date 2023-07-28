
import numpy

tolerances=dict(
    rtol=1e-3,
    atol=1e-2,
)

def assert_is_close(v1, v2, b, results=None, **tols):
    if results is None:
        results = dict(v1=v1, v2=v2)
    tols = {**tolerances, **tols}
    diff = numpy.subtract(v1, v2)
    assert (
        numpy.isclose(v1, v2, **tols).all()
    ) == b, dict(
        **results,
        diff=diff,
    )
