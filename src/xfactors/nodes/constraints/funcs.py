
import functools

import jax
import jax.numpy
import jax.numpy.linalg

# ---------------------------------------------------------------

def loss_mse(l, r):
    return jax.numpy.square(jax.numpy.subtract(l, r)).mean()

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

def loss_orthonormal(X, scale = 1.):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T) / scale
    return loss_mse(XXt, eye)

def loss_orthogonal(X, scale = 1.):
    XXt = jax.numpy.matmul(X, X.T) / scale
    return loss_diag(
        XXt, 
    )

# ---------------------------------------------------------------
