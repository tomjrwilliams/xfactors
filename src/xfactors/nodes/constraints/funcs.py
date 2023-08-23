
import functools

import jax
import jax.numpy
import jax.numpy.linalg

from ... import xfactors as xf

# ---------------------------------------------------------------

def loss_mabse(l, r):
    return jax.numpy.abs(jax.numpy.subtract(l, r)).mean()

def loss_mse(l, r):
    return jax.numpy.square(jax.numpy.subtract(l, r)).mean()

def loss_sumse(l, r):
    return jax.numpy.square(jax.numpy.subtract(l, r)).sum()

def loss_mse_zero(X1):
    return jax.numpy.square(X1).mean()

@functools.lru_cache(maxsize=4)
def loss_mean_zero(axis):
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    return f

# ascending just reverse order of xl and xr
def loss_descending(x):
    order = jax.numpy.flip(jax.numpy.argsort(x))
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()

def loss_diag(X):
    diag = jax.numpy.diag(X)
    diag = jax.numpy.multiply(
        jax.numpy.eye(X.shape[0]), diag
    )
    return loss_mse(X, diag)

# can be zero by col1 = col2 * -1
# so we first set all to have same sign (+) in col[0]
def loss_orthonormal(X):
    X = jax.numpy.multiply(
        X, xf.expand_dims(jax.numpy.sign(X[0]), 0, X.shape[0])
    )
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T)
    return loss_mse(XXt, eye)

def loss_orthogonal(X):
    XXt = jax.numpy.matmul(X, X.T)
    return loss_diag(XXt)

# the problem with straight eigval max
# - the eigval term dominates the orthogonality unit norm, especially for larger
# - two can be orthogonal by *-1 so duplicate the largest (again, dominating orth norm)
# fix for 1: scale by unit norm (and maximise)
# fix for 2: minimise the cross term
# by clamping norm, pushes eigval to be bigger, rather than w to be beyond unit
# https://proceedings.neurips.cc/paper_files/paper/2019/file/7dd0240cd412efde8bc165e864d3644f-Paper.pdf
def loss_eigenvec(cov, w, eigvals):
    cov_w = jax.numpy.matmul(cov, w)
    w_scale = jax.numpy.multiply(xf.expand_dims(eigvals, 0, 1), w)

    norm = jax.numpy.square(jax.numpy.matmul(w.T, w))

    mul = jax.numpy.clip(jax.numpy.multiply(
        norm, 
        1 + (
            jax.numpy.eye(norm.shape[0]) * -2
        ),
    ), a_max = 1., a_min = -1.)

    return (
        loss_mse(cov_w, w_scale)
        + jax.numpy.matmul(mul, eigvals).sum()
    )

# ---------------------------------------------------------------
