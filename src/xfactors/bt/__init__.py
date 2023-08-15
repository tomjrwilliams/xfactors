from . import strategies
from . import weights


# the factors are only useful
# to the extent that portfolios weighted by the factor


# either trends
# so after a certain accumulation cutoff at a certain window
# is more likely to continue


# or reverts
# so the opposite - after a certain acc cutoff, within a certain window
# likely to revert


# where the simplest is cutoff = 0, window = 1d



# those are both directional signals on the return of the factor portfolio
# as whole


# there's also then single name portfolios
# formed by accing performance over a window per name
# and comparing to that implied by the factor


# and then saying does that delta (per single name)
# persist or revert

# ie. signle name, vs factor of which it's meaningfully related
# the above process - but on that delta instead
# the meanginfully related is key (delta from zero weight isn't informative as the factor doesn't have a prediction)
# ie. the factor needs to have a strong prediction to diff against



# or is useful for first order hedging away
# to a portfolio that behaves as above




# where the point vs single name for eg. trend
# is that there's a smoothing of the signal from the portfolio




# if the factor structure genuinely rolls throguh time
# as different idiosycnratic shocks propagate differently
# which might even make esne if previous ones are traded against
# maybe the answer is indeed rolling?


# possibly, can trade against new factors vs a reference history library
# but perhaps not