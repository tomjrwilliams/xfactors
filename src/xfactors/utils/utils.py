
import jax.numpy

# ---------------------------------------------------------------

# log = False
def calc_returns(
    prices,
):
    return

# def calc_returns_index
# 


# ---------------------------------------------------------------

def cov(data):
    return jax.numpy.cov(
        jax.numpy.transpose(data.values)
    )

def center_zscore(data):
    return


# ---------------------------------------------------------------


# convolution functions / wrappers (on scipy presumably)

# ie. one way of centering is a rolling i_beta
# on the price series

# which is calculatable with with an appropriate kernel
# eg. see notebooks.ibeta-kernel


# but interesting to also look at the other funcs in kernels


# the kernels are also used for static specs of the embedding functions
# ie. can refer to a kernel by name (opt params)

# eg. for linear, sigmoid, etc.


# ---------------------------------------------------------------


# also for building up the df_feature representations

# eg. given classification trees, other categorical features


# as the xfactors methods assume just numeric labelled dataframe (dicts)




# ---------------------------------------------------------------


# also utils for graph blocks of input data / distribution
# factors, factor paths, etc.

# or separate utils for those probably

# ---------------------------------------------------------------
