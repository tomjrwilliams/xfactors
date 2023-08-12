
# plot the derivatives as well of the below curves
# over the domain

# against each param (?)
import jax


def sigmoid_curve(x, upper = 1, mid = 0, rate = 1):
    return upper / (
        1 + jax.numpy.exp(-1 * rate * (x - mid))
    )

def overextension(x, mid = 0):
    return x * jax.numpy.exp(-(x - mid).square())

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