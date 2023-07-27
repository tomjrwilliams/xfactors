
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

def normal(shape, seed = 69):
    return jax.random.normal(
        next_key(seed=seed), 
        shape = tuple(shape),
        #
    )

def vnormal(shape, n, seed = 69):
    keys = next_keys(n, seed=seed)
    f = jax.vmap(functools.partial(
        jax.random.normal, 
        shape=tuple(shape),
        #
    ))
    return f(keys)

# ---------------------------------------------------------------
