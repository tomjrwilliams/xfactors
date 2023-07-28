


# weights per ticker, for factors

# held to be orthogonal etc.

# but themselves drawn from a multi v gaussian

# with a given param mean

# and a covar given by a gp kernel

# fit against a set of latent params, say dim=2
# for similiarity (of weights) of pairs of tickers



# ah so perhaps there is then a variance term at factor level?
# even if they sum to orthogonality per instance
# one factor might vary much more across population?

# and the covariance is shared across factors
# so need another scaling term for per factor variance




# can visualise by then eg. clustering
# and averaging weights per sector / index within clusters

# per factor (?)




# the thing with the above, is we need a gaussian per factor weighting

# they can all use the same covariance though?
# and then sample independently from it?

# as i guess we're assuming same scale for the weights


# ie. we assume similarity is the same between tickers
# across factor exposure