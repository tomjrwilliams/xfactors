
import typing
import datetime

import numpy
import pandas
import jax

import xtuples

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

@xtuples.nTuple.decorate
class PCA(typing.NamedTuple):

    weights: pandas.DataFrame = None
    eigenvalues: pandas.Series = None

    flags_nan: Flags_NaN = None
    # flag to cache factors on fit?

    def encode_data(self, data):
        return

    def decode_factors(self, factors):
        return

    encode = encode_data
    decode = decode_factors

    def fit(self, data: pandas.DataFrame):

        # data_type = dict(
        #     prices=prices,
        #     retns=retns,
        # )
        # assert len(list(
        #     v for v in data_type.values() if v
        #     #
        # )) == 1, data_type
        
        # above not needed?
        # whether price or return (ie. if need centering) etc. first
        # is not the concern of the factor decomp

        # return self.update(weights = weights)

        # and the factor path, eigenvalues

        # but the path is separate, don't cache?

        return

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class Flags_PPCA(typing.NamedTuple):
    """
    >>> print(Flags_PPCA())
    """
    ordered: bool = True
    orthogonal: bool = True

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class PPCA(typing.NamedTuple):

    weights: pandas.DataFrame = None
    eigenvalues: pandas.Series = None

    flags_nan: Flags_NaN = None
    flags_ppca: Flags_PPCA = None
    # flag to cache factors on fit?

    def encode_data(self, data):
        return

    def decode_factors(self, factors):
        return

    encode = encode_data
    decode = decode_factors

    # interesting idea of option for only fitting the factors
    # given a fixed set of weights
    # so eg. only store the weights
    # then re-fit the factor path

    def fit(self, data: pandas.DataFrame):

        # factor fit but via gradient descent

        # not robust for efficiency
        # as that makes indexing more complicated?

        return

# ---------------------------------------------------------------

@xtuples.nTuple.decorate
class PPCA_Robust(typing.NamedTuple):

    weights: pandas.DataFrame = None
    eigenvalues: pandas.Series = None

    flags_nan: Flags_NaN = None
    flags_ppca: Flags_PPCA = None
    # flag to cache factors on fit?

    def encode_data(self, data):
        return

    def decode_factors(self, factors):
        return

    encode = encode_data
    decode = decode_factors

    def fit(
        df: pandas.DataFrame,
        flags_nan: Flags_NaN,
        flags_ppca: Flags_PPCA,
        model=None,
        #
    ):
        # parametrise the orthogonality contraint?
        # ie. how strongly we enforce?

        # instead of pca, use alternating least squares
        # to least squares minimise the reconstructed path

        # each point in the path needs at least some values?
        # else completely unconstrained
        # max_na / min_values together limit which rows are dropped
        # from the result

        # this doesn't take fill_nan as we just don't put into optimiser

        # this and the above are still index based factors
        # the below is not

        return

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
    eigenvalues: pandas.Series = None

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
    eigenvalues: pandas.Series = None
    
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
    eigenvalues: pandas.Series = None
    
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
    eigenvalues: pandas.Series = None

@xtuples.nTuple.decorate
class Results_Kernel_Factors(typing.NamedTuple):
    weights: pandas.DataFrame
    kernels: pandas.DataFrame # ?

    # this is eg. then a dataframe of params
    # for eg. linear, quadratic, interpolated lookup, etc.
    # per factor

    factors: pandas.DataFrame
    eigenvalues: pandas.Series = None

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
