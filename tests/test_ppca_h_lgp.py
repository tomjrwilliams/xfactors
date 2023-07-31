

# similar idea to ppca gp


# hierarchical means separate mu and covar

# l in lgp means the similarity is over a fitten latent 
# similarity space

# so we can also have non hierarchical - single similarity space
# whereas hierarchical we first split on class




# but we fit a separate mean per (presumed non overlapping) label

# h being hierarchical


# so presumably also a separate variance on each of the factor terms
# per cluster




# and covariance is only within clusters




# can then visualise again by plotting the sector mean vs one another
# per factor

# or flipping, and plotting per factor, the mean loadings


# similarly, plotting the per sector variance in loadings


# and finalyl clustering within sector based on the latent gp embedding