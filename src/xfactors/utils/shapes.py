import functools
import jax

# ---------------------------------------------------------------

def expand_dims(v, axis, size):
    v_expand = jax.numpy.expand_dims(v, axis)
    if axis == -1:
        axis = len(v_expand.shape) - 1
    res = jax.numpy.tile(
        v_expand,
        tuple([
            *[1 for _ in v.shape[:axis]],
            size,
            *[1 for _ in v.shape[axis:]],
        ])
    )
    return res

def expand_dims_like(v, axis, like):
    v_expand = jax.numpy.expand_dims(v, axis)
    if axis == -1:
        axis = len(v_expand.shape) - 1
    return jax.numpy.tile(
        v_expand,
        tuple([
            *[1 for _ in v.shape[:axis]],
            like.shape[axis],
            *[1 for _ in v.shape[axis:]],
        ])
    )
    
unsqueeze = functools.partial(
    expand_dims, axis=0, size=1,
)

# ---------------------------------------------------------------
