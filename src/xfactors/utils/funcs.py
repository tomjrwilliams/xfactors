
import functools

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

from . import shapes

# ---------------------------------------------------------------

sq = jax.numpy.square
mm = jax.numpy.matmul

# ---------------------------------------------------------------


def loss_mabse(l, r):
    return jax.numpy.abs(jax.numpy.subtract(l, r)).mean()

def loss_mse(l, r):
    return sq(jax.numpy.subtract(l, r)).mean()

def loss_sumse(l, r):
    return sq(jax.numpy.subtract(l, r)).sum()

def loss_mse_zero(X1):
    return sq(X1).mean()

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

def to_diag(v):
    return jax.numpy.multiply(
        jax.numpy.eye(v.shape[0]), v
    )

def loss_diag(X):
    diag = jax.numpy.diag(X)
    diag = jax.numpy.multiply(
        jax.numpy.eye(X.shape[-1]), diag
    )
    return loss_mse(X, diag)

def match_sign_to(X, row = None, col = None):
    assert row is not None or col is not None
    # assert X.shape[0] == X.shape[1] ?
    if row is not None:
        assert col is None
        return jax.numpy.multiply(
            X, shapes.expand_dims(
                jax.numpy.sign(X[row]), 0, X.shape[0]
            )
        )
    if col is not None:
        assert row is None
        return jax.numpy.multiply(
            X, shapes.expand_dims(
                jax.numpy.sign(X[..., col]), 1, X.shape[1]
            )
        )

def loss_orthonormal(X):
    eye = jax.numpy.eye(X.shape[-1])
    XXt = mm(X, shapes.transpose(X))
    return loss_mse(XXt, eye)

def loss_orthogonal(X):
    XXt = mm(X, shapes.transpose(X))
    return loss_diag(XXt)


# https://proceedings.neurips.cc/paper_files/paper/2019/file/7dd0240cd412efde8bc165e864d3644f-Paper.pdf
def loss_eigenvec(cov, w, eigvals):
    cov_w = mm(cov, w)
    w_scale = jax.numpy.multiply(shapes.expand_dims(eigvals, 0, 1), w)

    _mse = loss_mse(cov_w, w_scale)
    _norm = loss_eigenvec_norm(w, eigvals)

    return _mse + _norm

# NOTE: assumes eigvals already positive constrained
def loss_eigenvec_norm(w, eigvals):
    norm = mm(shapes.transpose(w), w)
    norm_sq = sq(norm)

    # norm_unit = jax.nn.sigmoid(norm)
    # norm_unit = jax.numpy.tanh(norm_sq)
    # norm_unit = (2 / (1 + jax.numpy.exp(-norm_sq))) - 1
    norm_unit = jax.numpy.clip(norm_sq, a_max=1.)

    mul = jax.numpy.multiply(
        norm_unit, 
        1 + (
            jax.numpy.eye(norm.shape[-1]) * -2
        ),
    )

    # if len(eigvals.shape) > 1:
    #     eigsum = eigvals.sum(axis=0)
    # else:
    #     eigsum = eigvals.sum()

    return mm(mul, eigvals).mean()


# ---------------------------------------------------------------

def diffs_1d(data):
    data_ = shapes.expand_dims(data, 0, 1)
    return data_ - data_.T

def diff_euclidean(l, r, small = 10 ** -3):
    diffs_sq = sq(jax.numpy.subtract(l, r))
    return jax.numpy.sqrt(
        jax.numpy.sum(diffs_sq, axis = -1) + small
    )

def diff_mahalanobis(
    data,
    mu,
    cov = None,
    cov_inv = None,
    sqrt=True,
):
    assert cov is not None or cov_inv is not None
    if cov_inv is not None:
        assert cov is None
    else:
        cov_inv = jax.numpy.linalg.inv(cov)
    mu_diff = data - mu

    if len(mu_diff.shape) == 3:
        lhs = jax.numpy.transpose(
            mu_diff,
            (0, 2, 1),
        )
        rhs = mu_diff
    else:
        lhs = mu_diff.T
        rhs = mu_diff

    res = mm(mm(lhs, cov_inv), rhs)
    if sqrt:
        return jax.numpy.sqrt(res)
    return res

# ---------------------------------------------------------------

small = 10 ** -4

def normalisation_gaussian(cov):
    det = jax.numpy.linalg.det(cov)
    rhs = jax.numpy.power(2 * numpy.pi, cov.shape[-1])
    det_rhs = det * rhs
    res = 1 / jax.numpy.sqrt(det_rhs)
    return res

def likelihood_gaussian(data, mu, cov):
    norm = normalisation_gaussian(cov)
    prob_unnorm = jax.numpy.exp(
        (-1 / 2) * diff_mahalanobis(data, mu, cov = cov, sqrt=False)
    )
    return prob_unnorm * norm

def log_likelihood_gaussian(data, mu, cov):
    norm = normalisation_gaussian(cov)
    prob_unnorm = (
        (-1 / 2) * diff_mahalanobis(data, mu, cov = cov, sqrt=False)
    )
    return prob_unnorm + jax.numpy.log(norm)

# ---------------------------------------------------------------

# (batch), n_dims

def normalisation_gaussian_diag(eigvals):
    res = 1 / jax.numpy.sqrt(
        (2 * numpy.pi) * eigvals
    )
    # assert not jax.numpy.isnan(res).any()
    # assert (jax.numpy.abs(res) < 10 ** 5).all()
    return res

def likelihood_gaussian_diag(data, mu, eigvals, eigvecs):
    norm = normalisation_gaussian_diag(eigvals)
    y = mm(eigvecs, data - mu)
    y_sq = sq(y)
    # := (batch), n_dims
    un_norm = jax.numpy.exp(- y_sq / (2 * eigvals))
    prob = norm * un_norm
    return jax.numpy.product(prob, axis=-1)

def log_likelihood_gaussian_diag(data, mu, eigvals, eigvecs):
    # assert not (eigvals <= 0).any()
    norm = normalisation_gaussian_diag(eigvals)
    # assert not jax.numpy.isnan(norm).any()
    y = mm(eigvecs, data - mu)
    y_sq = sq(y)
    un_norm = jax.numpy.exp(- y_sq / (2 * eigvals))
    # assert not jax.numpy.isnan(norm * un_norm).any(), [
    #     eigvals,
    # ]
    log_prob = jax.numpy.log(norm * un_norm)
    # assert not jax.numpy.isnan(log_prob).any()
    return jax.numpy.sum(log_prob, axis=-1)


# ---------------------------------------------------------------

# sigma ie. standard deviation
def alpha_beta_steady_state_kalman_params(
    sigma_process, sigma_noise, T = 1
):
    l = (sigma_process * sq(T)) / sigma_noise
    r = (
        4 + l - jax.numpy.sqrt((8 * l) + sq(l))
    ) / 4
    alpha = 1 - sq(r)
    beta = (2 * (2 - alpha)) - (4 * jax.numpy.sqrt(1 - alpha))
    return alpha, beta

ab_kalman_params = alpha_beta_steady_state_kalman_params

def alpha_beta_steady_state_residual_variance(
    sigma_noise, alpha
):
    return sq(sigma_noise) / (
        1 - sq(alpha)
    )

# ---------------------------------------------------------------

def linear(data, a, b):
    return (data * b) + a

def expit(data):
    return jax.scipy.special.expit(data)

def logistic(data):
    return 1 / (
        jax.numpy.exp(data) + 2 + jax.numpy.exp(-data)
    )

kernel_logistic = logistic

def sigmoid(data):
    return (2 / numpy.pi) * (
        1 / (jax.numpy.exp(data) + jax.numpy.exp(-data))
    )

kernel_sigmoid = sigmoid

def kernel_cosine(data):
    return (numpy.pi / 4) * (
        jax.numpy.cos(
            (numpy.pi / 2) * expit(data)
        )
    )

def rbf(data, l = 1., sigma = 1.):

    data_sq = sq(data)
    sigma_sq = sq(sigma)

    l_sq_2 = 2 * sq(l)

    return jax.numpy.exp(
        -1 * (data_sq / l_sq_2)
    ) * sigma_sq

kernel_rbf = rbf

def rq(data, sigma = 1., l = 1., a = 0.):

    sigma_sq = sq(sigma)
    a_2_ls_sq = 2 * sq(l) * a

    data_sq = sq(data)

    return jax.numpy.power(
        1 + (data_sq / a_2_ls_sq),
        -a
    ) * sigma_sq

kernel_rq = rq

def kernel_gaussian(data, sigma = 1.):

    sigma_sq = sq(sigma)
    data_sq = sq(data)

    norm = 1 / (2 * sigma_sq)

    return jax.numpy.exp(
        1 - (norm * data_sq)
    )

def kernel_exponential(data, sigma = 1.):

    data_abs = jax.numpy.abs(data)

    norm = 1 / sigma

    return jax.numpy.exp(
        1 - (norm * data_abs)
    )

def laplacian(data, sigma = 1.):

    # because has to be positive
    sigma_sq = sq(sigma)
    data_sq = sq(data)

    return jax.numpy.exp(
        - sigma_sq * data_sq
    )

kernel_laplacian = laplacian

def cauchy(data, sigma = 1.):

    sigma_sq = sq(sigma)
    data_sq = sq(data)

    return 1 / (
        1 + (data_sq / sigma_sq)
    )

kernel_cauchy = cauchy

def triangular(data, sigma = 1.):
    
    data_abs = jax.numpy.abs(data)

    return jax.numpy.clip(
        1 - (data_abs / (2 * sigma)),
        a_min=0.,
    )

kernel_triangular = triangular

def kernel_ou(data, sigma):

    data_sq = sq(data)
    
    return jax.numpy.exp(
        (- data_sq) / sigma
    )

# ---------------------------------------------------------------

def sigmoid_curve(x, upper = 1, mid = 0, rate = 1):
    return upper / (
        1 + jax.numpy.exp(-1 * rate * (x - mid))
    )

def overextension(x, mid = 0):
    return x * jax.numpy.exp(
        -1 * sq(x - mid)
    )

def overextension_df(df, mid=0):
    return pandas.DataFrame(
        overextension(df.values),
        columns=df.columns,
        index=df.index,
    )

# rate??
def gaussian(x, rate = 1, mid = 0):
    return 2 / (
        1 + jax.numpy.exp(rate * (x - mid).square)
    )

def gaussian_flipped(x, rate = 1, mid = 0):
    return 1 - gaussian(x, rate = rate, mid = mid)

def gaussian_sigmoid(x, rate = 1, mid = 0):
    return 1 + (-1 / (
        1 + jax.numpy.exp(-1 * rate * (x - mid).square)
    ))


# TODO: eg. gaussian surface for convolution kernels


def slope(x, rate = 1):
    return jax.numpy.log(1 + jax.numpy.exp(rate * x))

def trough(x, mid):
    return 1 / (
        1 + jax.numpy.exp(-x (x - mid))
    )


# ---------------------------------------------------------------

# hyperbolic tangent is an s curve

# of a falling object at time t
# is just the positive side, limiting to at terminal velocity
def velocity(
    t,
    mass,
    gravity,
    drag,
    density,
    area,
    g=10,
):
    alpha_square = (
        density * area * density,
    ) / (2 * mass * gravity)
    #
    alpha = alpha_square ** (1/2)
    #
    return (1 / alpha) * jax.numpy.tanh(
        alpha * g * t
    )


# sech(x) looks a bit like a gaussian
# = 2 / (e^x + e^-x)

# tanh(x) s curve
# = (e^x - e^-x ) / (e^x + e^-x )
# = (e^2x - 1) / (e^2x + 1)

# ---------------------------------------------------------------
