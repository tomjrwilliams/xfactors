
import functools

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
from jax.config import config 
# config.update("jax_debug_nans", True) 

def update_param(param, grad, lr = 0.1):
    assert grad.shape == param.shape, grad.shape
    assert not jax.numpy.isnan(grad).any()
    res = param - lr * grad
    assert not jax.numpy.isinf(res).any()
    return res

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

# @jax.jit
def loss_mse(X1, X2, scale = 1.):
    return jax.numpy.square(
        jax.numpy.subtract(X1, X2) * scale
    ).mean()

# @jax.jit
def zero_mask(X, should_mask):
    return jax.numpy.multiply(X, 1 + (-1 * should_mask))

# @jax.jit
def loss_mse_masked(X1, X2, should_mask):
    diff = jax.numpy.subtract(X1, X2)
    return jax.numpy.square(zero_mask(diff, should_mask)).mean()

# @jax.jit
def loss_mse_zero(X1):
    # X2 = jax.numpy.zeros(X1.shape)
    return jax.numpy.square(X1).mean()
    # return loss_mse(X1, X2)

@functools.lru_cache(maxsize=4)
def loss_mean_zero(axis):
    # @jax.jit
    def f(X):
        return loss_mse_zero(X.mean(axis=axis))
    return f

# @jax.jit
def loss_descending(x):
    order = jax.numpy.flip(jax.numpy.argsort(x))
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xl, xr).mean()
    # return -1 * jax.numpy.log(jax.numpy.divide(xl, xr)).mean()

# @jax.jit
def loss_abs_descending(x):
    x = jax.numpy.abs(x)
    return loss_descending(x)

# @jax.jit
def loss_ascending(x):
    order = jax.numpy.argsort(x)
    x_sort = x[order]
    acc = jax.numpy.cumsum(jax.numpy.flip(x_sort))
    xl = x_sort[..., :-1]
    xr = acc[..., 1:]
    return -1 * jax.numpy.subtract(xr, xl).mean()

# @jax.jit
def loss_abs_ascending(x):
    x = jax.numpy.abs(x)
    return loss_ascending(x)

# @jax.jit
def loss_orthogonal(X, scale = 1.):
    eye = jax.numpy.eye(X.shape[0])
    XXt = jax.numpy.matmul(X, X.T) / scale
    # return jax.numpy.square(jax.numpy.subtract(XXt, eye)).mean()
    return loss_mse(XXt, eye)


# @jax.jit
def loss_unit_norm(X):
    X2 = jax.numpy.sum(jax.numpy.square(X), axis = 0)
    return jax.numpy.square(X2 - 1).mean()

# @jax.jit
def loss_cov_eye(cov):
    eye = jax.numpy.eye(cov.shape[0])
    # return jax.numpy.square(jax.numpy.subtract(cov, eye)).mean()
    return loss_mse(cov, eye)

# @jax.jit
def loss_cov_diag(cov, diag):
    diag = jax.numpy.multiply(
        jax.numpy.eye(cov.shape[0]), diag
    )
    # return jax.numpy.square(jax.numpy.subtract(cov, diag)).mean()
    return loss_mse(cov, diag)

# ---------------------------------------------------------------

# @jax.jit
def encode_features_iw(data, feature_weights):
    if isinstance(data, pandas.DataFrame):
        return jax.numpy.matmul(data.values, feature_weights)
    elif isinstance(data, jax.numpy.ndarray):
        return jax.numpy.matmul(data, feature_weights)
    elif isinstance(data, numpy.ndarray):
        return jax.numpy.matmul(data, feature_weights)
    else:
        assert False, type(data)

def encode_features(model, features):
    if isinstance(model, PPCA_Instr_Weights):
        return encode_features_iw(features, model.feature_weights)
    assert False, model

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
    if isinstance(factors, jax.numpy.ndarray):
        return jax.numpy.matmul(factors, weights.T)
    elif isinstance(factors, numpy.ndarray):
        return jax.numpy.matmul(factors, weights.T)
    elif isinstance(factors, pandas.DataFrame):
        return jax.numpy.matmul(factors.values, weights.T)
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

def init_ppca_weights(data, n, orthogonal = False):
    if isinstance(data, int):
        shape = (data, n,)
    elif isinstance(data, tuple):
        shape = (data[1], n,)
    else:
        shape = (data.shape[1], n,)
    # for w in utils.uniform_spherical(shape, 1):
    #     return w
    if orthogonal:
        for w in utils.random_orthogonal(shape, 1):
            w_norm = w
            break
    else:
        for w in utils.random_normal(shape, 1):
            norm = jax.numpy.sqrt(
                jax.numpy.sum(jax.numpy.square(w), axis=0)
            )
            w_norm = jax.numpy.divide(w, norm)
            break
    if isinstance(data, (int, tuple)):
        return w_norm
    factors = jax.numpy.matmul(data, w_norm)
    cov = jax.numpy.cov(factors.T)
    eigvals = jax.numpy.diag(cov)
    order = jax.numpy.argsort(eigvals)
    return w_norm.T[jax.numpy.flip(order)].T

def init_ppca_eigvals(data, weights):
    factors = encode_data_pca(data, weights)
    return jax.numpy.log(jax.numpy.var(factors.T, axis = 1))

@jax.jit
def loss_ppca(weights, data):
    factors = jax.numpy.matmul(data, weights) # encode
    preds = jax.numpy.matmul(factors, weights.T) # decode
    cov = jax.numpy.cov(factors.T)
    eigvals = jax.numpy.diag(cov)
    return (
        - jax.numpy.product(1 + eigvals[:-1])
        + loss_descending(eigvals)
        + loss_orthogonal(weights.T)
        + loss_mean_zero(0)(factors)
        + loss_cov_diag(cov, eigvals)
        + loss_mse(data, preds)
    )

grad_loss_ppca = jax.jacrev(loss_ppca)

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

    # presumably a way of iteratively adding dims>
    # in the outer fit loop?
    # or incrementally change a weight decay parameter that is ramped down for the later dims as they come into play

    @classmethod
    def update(cls, weights, data, lr = 0.01):
        grad = grad_loss_ppca(weights, data)
        weights = update_param(weights, grad, lr = lr)
        return weights

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

@xtuples.nTuple.decorate
class Feature(typing.NamedTuple):

    name: str

    boolean: bool = False
    one_hot: bool = False

def build_feature_map(feature, df, tickers, index):
    if feature.boolean:
        return feature
    if feature.one_hot:
        # add one hot mapping table
        assert False, feature
    assert False, feature

# return shape = [n_index, n_tickers, n_vals]
def apply_feature_map(feature, df, tickers, index):
    if feature.boolean:
        # n_vals = 1
        return numpy.expand_dims(df.values, -1).astype(float)
    if feature.one_hot:
        # n_vals = arity(feature)
        assert False, feature
    assert False, feature

from jax.experimental import sparse

def init_weight_instruments(df, raw_features, n):

    tickers = xtuples.iTuple(df.columns)
    
    index = df.index

    for f_df in raw_features.values():
        if isinstance(f_df, pandas.DataFrame):
            index = index.union(f_df.index)
    
    raw_features = {
        feature: pandas.DataFrame({
            ticker: pandas.Series(
                index=index,
                data=(
                    [f_df[ticker] for _ in index]
                    if isinstance(f_df, pandas.Series)
                    else f_df[ticker].get(index, numpy.NaN)
                )
            )
            for ticker in tickers
        })
        for feature, f_df in raw_features.items()
    }

    feature_map = {
        feature.name: build_feature_map(
            feature, f_df, tickers, index
        )
        for feature, f_df in raw_features.items()
    }

    feature_vals = {
        feature: apply_feature_map(
            feature_map[feature.name],
            f_df,
            tickers,
            index,
        )
        for feature, f_df in raw_features.items()
    }

    features = jax.numpy.stack(
        list(feature_vals.values()),
        axis=-1
    )
    # shape = [n_index, n_tickers, sum(n_vals)]

    features = features.reshape(
        -1, features.shape[-1]
    )
    # := n_index * n_tickers, sum(n_vals)

    l = len(index)
    factors = jax.numpy.stack(list(
        utils.random_normal((l,), n)
    ))

    # later custom init
    # to go with the custom loss
    # eg. if some are independent of others, etc.
    weights_features = init_ppca_weights(features, n)

    # watch for NAs

    data = pandas.DataFrame({
        ticker: pandas.Series(
            index=index,
            data=df[ticker].get(index, numpy.NaN)
        )
        for ticker in tickers
    }).values

    data = data.reshape(-1)
    # := n_index * n_tickers, (1)

    i_map = jax.numpy.concatenate([
        jax.numpy.arange(len(index)) for _ in tickers
    ])

    features_na = numpy.sum(numpy.isnan(features), axis=1)
    data_na = numpy.isnan(data)

    not_na = numpy.add(features_na, data_na) == 0
    has_data = numpy.sum(numpy.abs(features), axis = 1) > 0

    keep_v = numpy.multiply(not_na, has_data) > 0

    order = numpy.argsort(i_map)
    keep_v = keep_v[order]
    
    keep = numpy.nonzero(keep_v)[0]
    assert len(keep.shape) == 1, keep.shape

    features = features[order][keep]
    data = data[order][keep]
    inds = i_map[order][keep]

    inds_onehot = numpy.zeros((data.shape[0], len(
        numpy.unique(inds)
    ),))
    for i, index in enumerate(inds):
        inds_onehot[i][index] = 1.

    assert (numpy.sum(inds_onehot, axis=1) == 1).all()

    assert features.shape[0] > 0, dict(
        data_na=data_na,
        features_na=features_na,
    )
    assert features.shape[0] == data.shape[0], dict(
        features=features.shape,
        data=data.shape,
    )
    assert (numpy.sum(features, axis = 1) > 0).all()

    # ind_counts = jax.ops.segment_sum(
    #     numpy.ones_like(inds), inds
    # )
    # ind_counts = numpy.array(ind_counts)

    return (
        data,
        factors,
        inds,
        inds_onehot,
        weights_features,
        features,
        feature_map,
    )

# later on make this use the feature map
# to return a custom loss
# eg. adding custom orthogonality / zeroing
# constraints across features

# data = n_indices, n_tickers
# features = n_indices * n_tickers, sum(n_vals)
# weights_features = sum(n_vals), n_features
# weights_factors = n_features, 1

# @jax.jit
def ppca_instr_weights_eigvals(
    w_features, w_factors, data, features, inds_onehot
):
    # w_features, w_factors = weights
    # norm inds?

    data_index = jax.numpy.multiply(
        utils.unsqueeze(data),
        inds_onehot,
    )
    # (n_samples, 1), (n_samples, n_indices)

    has_values = jax.numpy.count_nonzero(
        data_index, 
        axis = 0
    )

    features_w_features = jax.numpy.matmul(
        features,
        w_features,
    )
    # instr_factors = jax.numpy.multiply(
    #     utils.unsqueeze(data),
    #     features_w_features
    # )
    # n_samples, n_factors

    instr_factors = jax.numpy.matmul(
        data_index.T,
        features_w_features
    )
    # (n_samples, n_indices).T * (n_samples, n_factors)
    # = n_indices, n_factors

    # instr_factors = instr_factors[keep_indices, ...]

    instr_factors = jax.numpy.multiply(
        instr_factors,
        1 / utils.unsqueeze(has_values, -1)
    )
    # n_indices, n_factors

    mean_factors = jax.numpy.matmul(
        inds_onehot, instr_factors, 
    )
    # n_samples, n_indices * n_indices, n_factors
    # := n_samples, n_factors
    mean_factors = jax.numpy.multiply(
        utils.unsqueeze(w_factors.squeeze(), 0),
        mean_factors
    )

    # here we have aggregated weights per index
    # split back out to the relevant sample

    # so - now need, what, weights per sample
    # given features, from factors back to return?

    
    # instr_features = jax.numpy.matmul(
    #     # utils.unsqueeze(mean_factors, axis = 1), 
    #     mean_factors,
    #     w_features.T
    # )
    # n_indices, n_features

    # preds = jax.numpy.matmul(
    #     instr_features, 
    #     # jax.numpy.transpose(features, (0, 2, 1,))
    #     features.T
    # ).T
    # n_indices, n_samples

    preds = jax.numpy.sum(jax.numpy.multiply(
        mean_factors,
        features_w_features,
    ), axis=1)
    # n_samples, n_factors

    cov = jax.numpy.cov(instr_factors.T)
    eigvals = jax.numpy.diag(cov)

    eig_sort = jax.numpy.flip(jax.numpy.argsort(eigvals))

    print("instr_wwt", numpy.diag(numpy.matmul(instr_factors.T, instr_factors)))
    print("loss & preds (mu/var):", loss_mse(preds, data), numpy.mean(preds), numpy.var(preds))
    print("w_factors", w_factors.squeeze())
    # print("data (mu/var):", numpy.mean(data), numpy.var(data))

    # is covariance actually wrong
    # because we're already n_samples scaling with the has data?
    # so should just be mean_factors matmul?

    return jax.numpy.diag(cov)

@jax.jit
def loss_ppca_instr_weights(
    # weights_factors, 
    weights,
    data,
    features, 
    inds_onehot,
    mask,
    keep_indices,
    noise,
    # factor_mask,
):
    w_features, w_factors = weights
    # w_features, w_factors, factors = weights_factors
    # factors = jax.numpy.transpose(
    #     factors[:, factor_mask],
    #     axes=(1, 2, 0,)
    # )
    data = data[mask]
    features = features[mask]
    features = jax.numpy.add(features[mask], noise / 10)
    inds_onehot = inds_onehot[mask]

    data_index = jax.numpy.multiply(
        utils.unsqueeze(data),
        inds_onehot,
    )
    # (n_samples, 1), (n_samples, n_indices)

    has_values = jax.numpy.count_nonzero(
        data_index,
        axis = 0
    )
    data_index = data_index[..., keep_indices]
    has_values = has_values[keep_indices]

    features_w_features = jax.numpy.matmul(
        features,
        w_features,
    )
    # instr_factors = jax.numpy.multiply(
    #     utils.unsqueeze(data),
    #     features_w_features
    # )
    # n_samples, n_factors

    instr_factors = jax.numpy.matmul(
        data_index.T,
        features_w_features
    )
    # (n_samples, n_indices).T * (n_samples, n_factors)
    # = n_indices, n_factors

    instr_factors = instr_factors[keep_indices, ...]
    inds_onehot = inds_onehot[..., keep_indices]

    instr_factors = jax.numpy.multiply(
        instr_factors,
        1 / utils.unsqueeze(has_values, -1)
    )
    # n_indices, n_factors

    mean_factors = jax.numpy.matmul(
        inds_onehot, instr_factors, 
    )
    # n_samples, n_indices * n_indices, n_factors
    # := n_samples, n_factors
    mean_factors = jax.numpy.multiply(
        utils.unsqueeze(w_factors.squeeze(), 0),
        mean_factors
    )
    
    # instr_features = jax.numpy.matmul(
    #     # utils.unsqueeze(mean_factors, axis = 1), 
    #     mean_factors,
    #     w_features.T
    # )
    # n_indices, n_features

    # preds = jax.numpy.matmul(
    #     instr_features, 
    #     # jax.numpy.transpose(features, (0, 2, 1,))
    #     features.T
    # ).T
    # n_indices, n_samples

    preds = jax.numpy.sum(jax.numpy.multiply(
        mean_factors,
        features_w_features,
    ), axis=1)
    # n_samples, n_factors

    cov = jax.numpy.cov(instr_factors.T)
    eigvals = jax.numpy.diag(cov)

    eig_sort = jax.numpy.flip(jax.numpy.argsort(eigvals))

    loss = (
        # - jax.numpy.product(1 + eigvals[eig_sort][:-1])
        # + loss_descending(eigvals)
        # + loss_orthogonal(w_factors.T)
        # + loss_orthogonal(w_features.T)
        # + loss_orthogonal(instr_factors.T)
        + loss_mean_zero(0)(instr_factors)
        + loss_cov_eye(cov)
        + loss_mse(data, preds)
    )
    # assert not jax.numpy.isinf(loss)
    return loss

# the problem with the covariance
# is that a bunch of the samples are from the same day
# does it make more sense to take the covariance over the instr_factors?
# not with the prices icnluded

grad_loss_ppca_instr_weights = jax.jacrev(loss_ppca_instr_weights)

# todo: rename instr features

@xtuples.nTuple.decorate
class PPCA_Instr_Weights(typing.NamedTuple):

    # data columns specifically
    columns: xtuples.iTuple
    feature_map: xtuples.iTuple
    eigvals: pandas.Series
    weights_features: pandas.DataFrame
    weights_factors: pandas.DataFrame

    # flags_nan: Flags_NaN = None

    encode_features = encode_features
    encode_data = encode_data
    decode_factors = decode_factors

    # encode = encode_data
    # decode = decode_factors

    weights_df = weights_df

    @classmethod
    def update(
        cls,
        w_features,
        w_factors,
        # factors,
        data,
        features,
        inds_onehot,
        mask,
        keep_indices,
        noise,
        # factor_mask,
        lr = 0.01
    ):
        (
            grad_w_features,
            grad_w_factors,
            # grad_factors,
        ) = grad_loss_ppca_instr_weights(
            # w_features,
            (w_features, w_factors,),
            data,
            features,
            inds_onehot,
            mask,
            keep_indices,
            noise,
            # factor_mask,
            # feature_map
        )
        w_features = update_param(
            w_features, grad_w_features, lr = lr
        )
        w_factors = update_param(
            w_factors, grad_w_factors, lr = lr
        )
        # factors = update_param(
        #     factors, grad_factors, lr = lr
        # )
        return w_features, w_factors
        # factors

    @classmethod
    def fit(
        cls,
        df: pandas.DataFrame,
        features_raw: dict[str, typing.Union[
            pandas.Series,
            pandas.DataFrame,
        ]],
        n = 1,
        noise=1,
        iters = 2500,
        lr = 0.1,
        #
    ):
        (
            data,
            factors,
            inds,
            inds_onehot,
            weights_features, 
            features,
            feature_map,
        ) = init_weight_instruments(
            df, features_raw, n
        )

        # ind_map is groups of indices of data / features
        # with same index

        weights_factors = init_ppca_weights(
            n, 1
        )

        print("Shapes:")
        print(dict(
            weights_features=weights_features.shape,
            # weights_factors=weights_factors.shape,
            data=data.shape,
            features=features.shape,
            # factors=factors.shape,
        ))

        # given fixed batch size, set prob threshold
        # as ratio of len(data)

        # n_indices = len(ind_counts)
        # n_values = 5

        # cum_counts = numpy.concatenate([
        #     numpy.array([0]),
        #     numpy.cumsum(ind_counts)
        # ])[:-1]
        # # ind_map = [
        # #     (numpy.arange(l) + prev).astype(int)
        # #     for l, prev in zip(ind_counts, cum_counts)
        # # ]
        # maxl = max(ind_counts)

        # def index_mask(inds, n_samples):
        #     inds = numpy.array(inds)
            
        #     counts = ind_counts[inds]
        #     ind_prev = cum_counts[inds]

        #     ind_sample = xtuples.iTuple(list(
        #         utils.random_choices(
        #             numpy.arange(maxl),
        #             (n_samples,),
        #             len(inds),
        #         )
        #     )).map(numpy.array)

        #     mask = ind_sample.zip(counts, ind_prev).mapstar(
        #         lambda sample, l, prev: numpy.mod(
        #             sample, l
        #         ) + prev
        #     ).pipe(list)
            
        #     return numpy.vstack(mask).astype(int)

        # mask = index_mask(numpy.arange(len(ind_counts)), maxl)
        # eigvals = ppca_instr_weights_eigvals(
        #     weights_features, 
        #     weights_factors, data[mask], features[mask]
        # )

        # order = numpy.flip(numpy.argsort(eigvals))

        # weights_features = weights_features[..., order]
        # weights_factors = jax.numpy.sqrt(eigvals[..., order])

        # rand_inds = list(
        #     utils.random_indices(
        #         len(ind_counts), (n_indices,), iters,
        #     )
        # )
        # print("Mapping index masks:", len(rand_inds))
        # masks = xtuples.iTuple(rand_inds).map(
        #     functools.partial(index_mask, n_samples=n_values)
        # )
        # print("Masks:", len(masks))

        data_index = jax.numpy.multiply(
            utils.unsqueeze(data),
            inds_onehot,
        )
        # (n_samples, 1), (n_samples, n_indices)

        # keep_indices = has_values > 0
        n_samples = 2500
        for iter, (mask, noise) in enumerate(zip(
            # masks,
            utils.random_indices(
                data.shape[0], (n_samples,), iters,
            ),
            utils.random_normal(
                (n_samples, features.shape[1],), iters,
            )
        )):
            # mask = index_mask(inds, n_values)
            # factor_mask = jax.numpy.resize(
            #     utils.unsqueeze(inds),
            #     (n_indices, n_values,)
            # )
            has_values = jax.numpy.count_nonzero(
                data_index[mask, ...], axis = 0
            )
            keep_indices = jax.numpy.nonzero(has_values)[0]

            if iter == 0:
                print("Sample shapes:")
                print(dict(
                    data=data[mask].shape,
                    features=features[mask].shape,
                    # factors=factors[:, factor_mask].shape,
                ))
            
            (
                weights_features,
                weights_factors,
                # factors,
            ) = cls.update(
                weights_features,
                weights_factors,
                # factors,
                data,
                features,
                inds_onehot,
                mask,
                keep_indices,
                noise,
                # factor_mask,
                lr = lr
            )

            if iter % (iters / 10) == 0 or iter == iters - 1:
                # eigvals = numpy.diag(numpy.cov(factors))
                # mask = index_mask(numpy.arange(len(ind_counts)), maxl)
                eigvals = ppca_instr_weights_eigvals(
                    weights_features, 
                    weights_factors,
                    data,
                    features,
                    inds_onehot,
                )
                print(iter, ":", iters, eigvals)

        # keep = numpy.flip(numpy.argsort(eigvals))[:n]

        # eigvals = eigvals[..., keep]
        # weights_features = weights_features[..., keep]
        # weights_factors = weights_factors[..., keep, :]
    
        # apply_feature_map(...)?

        return cls(
            xtuples.iTuple(df.columns),
            feature_map,
            eigvals,
            weights_features,
            weights_factors,
        )
        
        # the kernel gives you weights
        # ie. features @ instrumnts = weights

        # that are then multiplied onto the factor path
        # to get the final return values
        # ie. (features @ instruments) @ factor_paths = results
        

        # so each ticker needs a full (?) set of features
        # at each step

        # to get that weight vector
        
        # but we don't need every ticker per date

# ---------------------------------------------------------------


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
