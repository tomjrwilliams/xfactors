

import jax.numpy


# ---------------------------------------------------------------

# log = False
def calc_returns(
    prices,
):
    return

# def calc_returns_index

# ---------------------------------------------------------------

# wrap up in a context manager

KEYS = {}

def random_keys(n, seed = 69):
    if seed not in KEYS:
        KEYS[seed] = jax.random.PRNGKey(seed)
    key = KEYS[seed]
    for _ in range(n):
        key, subkey = jax.random.split(key)
        KEYS[seed] = key
        yield subkey

def random_sample(n, f, seed = 69):
    for subkey in random_keys(n, seed=seed):
        yield f(subkey)

def random_uniform(shape, n):
    f = lambda subkey: jax.random.uniform(subkey, shape=shape)
    yield from random_sample(n, f)

def random_uniform_indices(shape, n, threshold):
    for probs in random_uniform(shape, n):
        mask = probs <= threshold
        yield mask

def random_choices(v, shape, n, p = None):
    f = lambda subkey: jax.random.choice(
        subkey, v, shape=shape, p=p
    )
    yield from random_sample(n, f)

def random_indices(l, shape, n, p = None):
    f = lambda subkey: jax.random.choice(
        subkey, jax.numpy.arange(l), shape=shape, p=p, replace=False
    )
    yield from random_sample(n, f)

def random_beta(shape, n, a, b):
    f = lambda subkey: jax.random.beta(subkey, a, b, shape = shape)
    yield from random_sample(n, f)

def random_normal(shape, n):
    f = lambda subkey: jax.random.normal(subkey, shape=shape)
    yield from random_sample(n, f)

def random_orthogonal(shape, n):
    f = lambda subkey: jax.random.orthogonal(
        subkey, 
        n=1,
        shape=shape,
    ).squeeze().squeeze()
    yield from random_sample(n, f)


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
