
import typing
import datetime

import numpy
import pandas

import jax
import jax.numpy
import jax.numpy.linalg

import xtuples

import utils

# # ---------------------------------------------------------------

# @xtuples.nTuple.decorate
# class Example(typing.NamedTuple):
#     x: float
#     y: float
    
#     update = xtuples.nTuple.update()

# def add_2(ex):
#     return ex.x + 2

# ex = Example(1., 2.)

# ex_grad = jax.grad(add_2)(ex)
# print(ex_grad)

# ex_val, ex_grad = jax.value_and_grad(add_2)(ex)
# print(ex_val, ex_grad)
    
# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Flags_NaN(typing.NamedTuple):
    """
    >>> print(Flags_NaN())
    """
    max_n: int = None
    min_values: int = None
    fill: float = None

def clean_df_nan(
    df: pandas.DataFrame,
    flags_nan: Flags_NaN,
    #
):
    return

# ---------------------------------------------------------------

# w = n_tickers, n_factors
def weights_df(model):
    return pandas.DataFrame(
        model.weights.T.real,
        columns=model.columns,
    )

# ---------------------------------------------------------------

# def meta_loss_fn(params, data):
#   """Computes the loss after one step of SGD."""
#   grads = jax.grad(loss_fn)(params, data)
#   return loss_fn(params - lr * grads, data)

# def update(theta, x, y, lr=0.1):
#   return theta - lr * jax.grad(loss_fn)(theta, x, y)

# meta_grads = jax.grad(meta_loss_fn)(params, data)

import jax.lax

@jax.jit
def loss_mse(X1, X2):
    vmax = 10. ** 5
    diffs = jax.numpy.square(jax.numpy.subtract(X1, X2))
    return diffs.mean()
    # jax.lax.clamp(
    #     -vmax,
    #     diffs,
    #     vmax,
    # ).mean()

@jax.jit
def zero_mask(X, should_mask):
    return jax.numpy.multiply(X, 1 + (-1 * should_mask))

@jax.jit
def loss_mse_masked(X1, X2, should_mask):
    diff = jax.numpy.subtract(X1, X2)
    return jax.numpy.square(zero_mask(diff, should_mask)).mean()

@jax.jit
def loss_mse_zero(X1):
    # X2 = jax.numpy.zeros(X1.shape)
    return jax.numpy.square(X1).mean()
    # return loss_mse(X1, X2)

def loss_mean_zero(axis, cls = False):
    @jax.jit
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    if not cls:
        return f
    def cls_f(cls, X):
        return f(X)
    return cls_f

@jax.jit
def loss_descending(x):
    acc = jax.numpy.cumsum(jax.numpy.flip(x))
    xl = x[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()
    # return -1 * jax.numpy.log(jax.numpy.divide(xl, xr)).mean()

@jax.jit
def loss_abs_descending(x):
    x = jax.numpy.abs(x)
    acc = jax.numpy.cumsum(jax.numpy.flip(x))
    xl = x[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()

@jax.jit
def loss_ascending(x):
    acc = jax.numpy.cumsum(jax.numpy.flip(x))
    xl = x[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xr, xl).mean()

@jax.jit
def loss_abs_ascending(x):
    x = jax.numpy.abs(x)
    acc = jax.numpy.cumsum(jax.numpy.flip(x))
    xl = x[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xr, xl).mean()

@jax.jit
def loss_orthogonal(X):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T)
    return jax.numpy.square(jax.numpy.subtract(XXt, eye)).mean()
    # return loss_mse(XXt, eye)


@jax.jit
def loss_unit_norm(X):
    X2 = jax.numpy.sum(jax.numpy.square(X), axis = 0)
    return jax.numpy.square(X2 - 1).mean()

@jax.jit
def loss_cov_eye(X):
    cov = jax.numpy.cov(X)
    eye = jax.numpy.eye(cov.shape[0])
    return jax.numpy.square(jax.numpy.subtract(cov, eye)).mean()
    return loss_mse(cov, eye)

@jax.jit
def loss_cov_diag(X, scales):
    cov = jax.numpy.cov(X)
    eye = jax.numpy.eye(cov.shape[0])
    diag = jax.numpy.multiply(eye, scales.T)
    return jax.numpy.square(jax.numpy.subtract(cov, diag)).mean()
    # return loss_mse(cov, diag)

@jax.jit
def loss_cov_invariance(eigvals, weights, data):
    lhs = jax.numpy.matmul(jax.numpy.cov(data.T), weights)
    eigvals = jax.numpy.expand_dims(eigvals, -1)
    rhs = jax.numpy.matmul(weights, eigvals)
    return jax.numpy.square(jax.numpy.subtract(lhs, rhs)).mean()

# ---------------------------------------------------------------

# @jax.jit
def encode_data_pca(data, weights):
    if isinstance(data, pandas.DataFrame):
        return jax.numpy.matmul(data.values, weights)
    elif isinstance(data, jax.numpy.ndarray):
        return jax.numpy.matmul(data, weights)
    elif isinstance(data, numpy.ndarray):
        return jax.numpy.matmul(data, weights)
    else:
        assert False, type(data)

def encode_data(model, data):
    if isinstance(model, PCA):
        return encode_data_pca(data, model.weights)
    if isinstance(model, PPCA):
        return encode_data_pca(data, model.weights)
    assert False, model

# ---------------------------------------------------------------

def clip_factors_pca(eigvals, weights, n):
    if n is None:
        return eigvals, weights
    return eigvals[..., :n], weights[..., :n]

# w = n_tickers, n_factors
# wT = n_factors, n_tickers
# @jax.jit
def decode_factors_pca(factors, weights):
    wT = jax.numpy.transpose(weights)
    if isinstance(factors, jax.numpy.ndarray):
        return jax.numpy.matmul(factors, wT)
    elif isinstance(factors, numpy.ndarray):
        return jax.numpy.matmul(factors, wT)
    elif isinstance(factors, pandas.DataFrame):
        return jax.numpy.matmul(factors.values, wT)
    else:
        assert False, type(factors)

# result = n_days, n_tickers
def decode_factors(model, factors):
    if isinstance(model, PCA):
        return decode_factors_pca(factors, model.weights)
    if isinstance(model, PPCA):
        return decode_factors_pca(factors, model.weights)
    assert False, model

# ---------------------------------------------------------------

# print(dict(
#     data=data.values.shape,
#     weights=weights.shape,
#     eigvals=eigvals.shape,
# ))

@xtuples.nTuple.decorate
class PCA(typing.NamedTuple):

    columns: xtuples.iTuple
    eigvals: pandas.Series
    weights: pandas.DataFrame

    # flags_nan: Flags_NaN = None

    encode_data = encode_data
    decode_factors = decode_factors

    encode = encode_data
    decode = decode_factors

    weights_df = weights_df

    @property
    def n(self):
        return len(self.eigvals)

    @classmethod
    def fit(cls, df: pandas.DataFrame, n = None):
        eigvals, weights = jax.numpy.linalg.eig(utils.cov(df))
        return cls(
            xtuples.iTuple(df.columns),
            *clip_factors_pca(eigvals, weights, n)
        )

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Flags_PPCA(typing.NamedTuple):
    """
    >>> print(Flags_PPCA())
    """
    ordered: bool = True
    orthogonal: bool = True

# ---------------------------------------------------------------

def init_ppca_weights(data, n):
    shape = (data.shape[1], n,)
    # for w in utils.uniform_spherical(shape, 1):
    #     return w
    # for w in utils.random_orthogonal(shape, 1):
    for w in utils.random_normal(shape, 1):
        norm = jax.numpy.sqrt(
            jax.numpy.sum(jax.numpy.square(w), axis=0)
        )
        return jax.numpy.divide(w, norm)

def init_ppca_eigvals(data, weights):
    factors = encode_data_pca(data, weights)
    return jax.numpy.log(jax.numpy.var(factors.T, axis = 1))

@xtuples.nTuple.decorate
class PPCA(typing.NamedTuple):

    columns: xtuples.iTuple
    eigvals: pandas.Series
    weights: pandas.DataFrame

    # flags_nan: Flags_NaN = None

    encode_data = encode_data
    decode_factors = decode_factors

    encode = encode_data
    decode = decode_factors

    weights_df = weights_df

    loss_mean_zero = classmethod(loss_mean_zero(0, cls = True))

    @classmethod
    def update_param(cls, param, grad, lr = 0.1):
        assert grad.shape == param.shape, grad.shape
        assert not jax.numpy.isnan(grad).any()
        return param - lr * grad

    @classmethod
    def loss_preds(cls, weights, data):
        factors = encode_data_pca(data, weights)
        preds = decode_factors_pca(factors, weights)
        return loss_mse(preds, data)

    @classmethod
    def update_preds(cls, eigvals, weights, data, lr = 0.01):
        grad = jax.jacrev(cls.loss_preds)(weights, data)
        weights = cls.update_param(weights, grad, lr = lr)
        return eigvals, weights

    @classmethod
    def loss_orthogonal(cls, weights):
        return loss_orthogonal(weights.T)

    @classmethod
    def update_orthogonal(cls, eigvals, weights, data, lr = 0.01):
        grad = jax.jacrev(cls.loss_orthogonal)(weights)
        weights = cls.update_param(weights, grad, lr = lr)
        return eigvals, weights

    @classmethod
    def loss(cls, weights, data):
        factors = encode_data_pca(data, weights)
        eigvals = jax.numpy.var(factors, axis=0)
        return (
            - jax.numpy.product(1 + eigvals[:-1])
            + loss_descending(eigvals)
            + loss_orthogonal(weights.T)
            + cls.loss_mean_zero(factors)
            + loss_cov_diag(factors.T, eigvals)
            + cls.loss_preds(weights, data)
        )

    @classmethod
    def update(cls, weights, data, lr = 0.01):
        grad = jax.jacrev(cls.loss)(weights, data)
        weights = cls.update_param(weights, grad, lr = lr)
        return weights

    # presumably a way of iteratively adding dims>
    # in the outer fit loop?
    # or incrementally change a weight decay parameter that is ramped down for the later dims as they come into play

    @classmethod
    def fit(
        cls,
        df: pandas.DataFrame,
        n = 1,
        noise=1,
        iters = 2500,
        lr = 0.1,
        #
    ):

        # plus null mask on mse(pred, data)
        # optional additional null mask df kwarg
        # eg. rolling index membership

        data = df.values

        weights = init_ppca_weights(data, n + noise)

        print(dict(
            weights=weights.shape,
            data=data.shape,
        ))

        for i in range(iters):
            weights = cls.update(
                weights, data, lr = lr
            )
            if i % (iters / 10) == 0 or i == iters - 1:
                eigvals = numpy.round(numpy.var(
                    encode_data_pca(data, weights), axis=0
                ), 3)
                print(i, ":", iters, eigvals)

        eigvals = eigvals[..., :n]
        weights = weights[..., :n]
    
        return cls(
            xtuples.iTuple(df.columns),
            eigvals,
            weights,
        )

# ---------------------------------------------------------------


# TODO: an even better way of doing it
# is to define each of the encode / decode functions as free function

# that we bind on class definition
# can even have a mixin version that we expose
# that one can inherit from

# and mixin, to form the combinations we want

# so the fit just calls self.decode_factors
# where the methods then need to have a consistent set of kwargs
# perhaps even wrapped up into

# data = ...
# weights(_kwargs) = ...
# factors(_kwargs) = ...


@xtuples.nTuple.decorate
class Instrumented_Weights(typing.NamedTuple):
    instruments: pandas.DataFrame
    factors: pandas.DataFrame
    eigvals: pandas.Series = None

    def encode_features(self, features):
        return

    def decode_factors(self, factors, weights):
        return

    def fit(
        df: pandas.DataFrame,
        dfs_features: dict[str, pandas.DataFrame],
        flags_nan: Flags_NaN,
        flags_nan_features: dict[str, Flags_NaN],
        flags_ppca: Flags_PPCA,
        model=None,
    ):

        # the kernel gives you weights
        # ie. features @ instrumnts = weights

        # that are then multiplied onto the factor path
        # to get the final return values
        # ie. (features @ instruments) @ factor_paths = results
        

        # so each ticker needs a full (?) set of features
        # at each step

        # to get that weight vector
        
        # but we don't need every ticker per date

        return

# TODO: as above, move to tuples
# that hold the parameters, though not by default the factor paths?


def variational_instrumented_weights(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # one approach would be to make all the empties
    # themselves parameters

    # and fit them at the same time as the factor_paths

    # or to parametrise just a distribution over them
    # ie. just a few parameters

    # and then sample the missin ones each period

    # ie. you treat the weights
    # as themselves like factor paths
    # lower dim representations (combinations) of the features
    # linear, invertible
    # so they're a combination of feature, but also can reconstruct the feature by just multiply by kernel.T

    # then fit a covariance structure to the weight distribution
    # so its sample-able from
    # and then you sample from the weight distribution
    # project up into the feature space
    # mse on the observed values

    return

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Results_Instruments_Factors(typing.NamedTuple):
    weights: pandas.DataFrame
    instruments: pandas.DataFrame
    eigvals: pandas.Series = None
    
def instruments_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # as per the above

    # but instead factor_path is instrumented

    # factor_paths = (features @ instruments)
    # ie. weights @ (features @ instruments) = results


    return

def variational_instrumented_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # similar idea to the other variational

    # but allowing for missing in the path feature sthis time

    # by fitting a generating distribution of the weights
    # and backfit on the projection back up to th efeatures

    # as well as the mse onto the results (eg. the yields or whatever)

    return

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Results_Instrumented(typing.NamedTuple):
    instruments_weights: pandas.DataFrame
    instruments_factors: pandas.DataFrame
    eigvals: pandas.Series = None
    
def instrumented_weights_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # this method should have compatible signature with both of the above
    # if none provided for relevant kwarg
    # pass through to the sub method

    # else, assumed both instrumented


    # as per the above

    # but instead factor_path is instrumented

    # factor_paths = (features @ instruments)
    # ie. weights @ (features @ instruments) = results

    return

def variational_instrumented_weights_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # similar idea to the other variational

    # but allowing for missing in the path feature sthis time

    # by fitting a generating distribution of the weights
    # and backfit on the projection back up to th efeatures

    # as well as the mse onto the results (eg. the yields or whatever)

    return

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Results_Kernel_Weights(typing.NamedTuple):
    kernels: pandas.DataFrame # ?

    # this is eg. then a dataframe of params
    # for eg. linear, quadratic, interpolated lookup, etc.
    # per factor

    factors: pandas.DataFrame
    eigvals: pandas.Series = None

@xtuples.nTuple.decorate
class Results_Kernel_Factors(typing.NamedTuple):
    weights: pandas.DataFrame
    kernels: pandas.DataFrame # ?

    # this is eg. then a dataframe of params
    # for eg. linear, quadratic, interpolated lookup, etc.
    # per factor

    factors: pandas.DataFrame
    eigvals: pandas.Series = None

def kernel_weights(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # instead of using index based factors
    # here, we use kernels

    # ( kernel(features) ) @ factor_paths = results

    # eg. grid of weights
    # and then a lookup from the kernel to the relevant weight cells
    
    # or, a slope function of the features:
    # weights = (features * beta) + intercept

    # or, quadratic

    # where each is specified independently, with appropriate constraints (eg. strictly positive, or only one param is positive, etc.)

    return


def kernel_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # weights @ ( kernel(features) ) = results

    return


def kernel_weights_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    # same idea as above
    # if either are none, pass through to relevant sub method

    # else, assume both kernel


    # weights @ ( kernel(features) ) = results

    return



# ---------------------------------------------------------------

def instrumented_weights_kernel_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    return

def instrumented_factors_kernel_weights(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    return

def variational_instrumented_weights_kernel_factors(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    return

def variational_instrumented_factors_kernel_weights(
    df: pandas.DataFrame,
    dfs_features: dict[str, pandas.DataFrame],
    flags_nan: Flags_NaN,
    flags_nan_features: dict[str, Flags_NaN],
    flags_ppca: Flags_PPCA,
    model=None,
):

    return

# ---------------------------------------------------------------

def decompose(
    df,
    # ...
):

    # master method
    # that takes specifications of factors and weights

    # and matches up to the relevant method from above

    # including straight pca / ppca

    return

# ---------------------------------------------------------------
