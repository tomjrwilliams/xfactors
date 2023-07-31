
import functools
import jax

# ---------------------------------------------------------------

# context generator for key scope

KEYS = {}

def next_key(seed = 69):
    if seed not in KEYS:
        KEYS[seed] = jax.random.PRNGKey(seed)
    KEYS[seed], subkey = jax.random.split(KEYS[seed])
    return subkey

def next_keys(n, seed = 69):
    if seed not in KEYS:
        KEYS[seed] = jax.random.PRNGKey(seed)
    KEYS[seed], *subkeys = jax.random.split(KEYS[seed], num=n)
    return jax.numpy.vstack(subkeys)

# ---------------------------------------------------------------

def gaussian(shape=None, mu=None, var=None, seed = 69):
    # if mu and cov one, need shape
    # else can infer shape from mu / cov
    # if shape given, assert matches mu / cov
    return jax.random.normal(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def v_gaussian(n, shape=None, mu=None, var=None, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.normal, 
        shape=tuple(shape),
        #
    ))
    return f(keys)

def mv_gaussian(shape=None, mu=None, cov=None, seed = 69):
    # if mu and cov one, need shape
    # else can infer shape from mu / cov
    # if shape given, assert matches mu / cov
    return jax.random.normal(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def v_mv_gaussian(shape, n, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.normal, 
        shape=tuple(shape),
        #
    ))
    return f(keys)

# ---------------------------------------------------------------

# mu can be vector, in which case we get correalted random walk
# summing over the (not necessarily diag cov) samples
def gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    shape = shape + (n,)
    v = gaussian(shape=shape, mu=mu, cov=cov, seed=seed)
    return jax.numpy.cumsum(v, dim = -1)

def v_gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    return

# geometric walk ie. cum prod not cumsum
def gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

def v_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

# mu can be vector, in which case we get correalted random walk
# summing over the (not necessarily diag cov) samples
def mv_gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    shape = shape + (n,)
    v = gaussian(shape=shape, mu=mu, cov=cov, seed=seed)
    return jax.numpy.cumsum(v, dim = -1)

def v_mv_gaussian_walk(n, shape=None, mu=None, var=None, seed=69):
    return

# geometric walk ie. cum prod not cumsum
def mv_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

def v_mv_gaussian_gwalk(n, shape=None, mu=None, var=None, seed=69):
    return

# ---------------------------------------------------------------

def orthogonal(shape, seed = 69):
    return jax.random.orthogonal(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def vorthogonal(shape, n, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.orthogonal, 
        shape=tuple(shape),
        #
    ))
    return f(keys)

# ---------------------------------------------------------------

# def random_sample(n, f, seed = 69):
#     for subkey in random_keys(n, seed=seed):
#         yield f(subkey)

# def random_uniform(shape, n):
#     f = lambda subkey: jax.random.uniform(subkey, shape=shape)
#     yield from random_sample(n, f)

# def random_uniform_indices(shape, n, threshold):
#     for probs in random_uniform(shape, n):
#         mask = probs <= threshold
#         yield mask

# def random_choices(v, shape, n, p = None):
#     f = lambda subkey: jax.random.choice(
#         subkey, v, shape=shape, p=p
#     )
#     yield from random_sample(n, f)

# def random_indices(l, shape, n, p = None):
#     f = lambda subkey: jax.random.choice(
#         subkey, jax.numpy.arange(l), shape=shape, p=p, replace=False
#     )
#     yield from random_sample(n, f)

# def random_beta(shape, n, a, b):
#     f = lambda subkey: jax.random.beta(subkey, a, b, shape = shape)
#     yield from random_sample(n, f)

# def random_normal(shape, n):
#     f = lambda subkey: jax.random.normal(subkey, shape=shape)
#     yield from random_sample(n, f)

# def random_orthogonal(shape, n):
#     f = lambda subkey: jax.random.orthogonal(
#         subkey, 
#         n=1,
#         shape=shape,
#     ).squeeze().squeeze()
#     yield from random_sample(n, f)

# # ---------------------------------------------------------------

# def norm_gaussian(v, n_sample_dims = None):
#     if n_sample_dims is None:
#         n_sample_dims = len(v.shape)
#     dims = tuple(range(len(v.shape)))[-n_sample_dims:]
#     mag = jax.numpy.sqrt(jax.numpy.sum(jax.numpy.square(v), dims))
#     for _ in range(n_sample_dims):
#         mag = jax.numpy.expand_dims(mag, -1)
#     mag = jax.numpy.resize(mag, v.shape)
#     return jax.numpy.divide(v, mag)

# def uniform_spherical(shape, n = 1):
#     for sample in random_normal(shape, n):
#         yield norm_gaussian(sample)

# # ---------------------------------------------------------------
