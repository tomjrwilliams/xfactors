
import functools
import jax

KEYS = {}

def random_keys(n, seed = 69):
    if seed not in KEYS:
        KEYS[seed] = jax.random.PRNGKey(seed)
    key = KEYS[seed]
    KEYS[seed], *subkeys = jax.random.split(key, num=n)
    return jax.numpy.vstack(subkeys)

def random_normal(shape, n, seed = 69):
    f = functools.partial(jax.random.normal, shape = shape)
    f_map = jax.vmap(f)
    keys = random_keys(n, seed=seed)
    return f_map(keys)