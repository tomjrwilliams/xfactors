# xfactors

[![PyPI - Version](https://img.shields.io/pypi/v/xfactors.svg)](https://pypi.org/project/xfactors)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xfactors.svg)](https://pypi.org/project/xfactors)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

TBC:

```console
pip install xfactors
```

## Overview

xf, short for xfactors (name to be reconsidered), is a work-in-progress library for practical machine learning research in finance built on [JAX](https://jax.readthedocs.io/en/latest/index.html).

xf provides:

- base classes / interfaces for defining re-useable JAX model components (as nodes in directed acyclic computation graphs).

- an apparatus for specifying how data, parameters, and intermediate results should flow such a model.

- a simple API for model training and application

Where we obey, as much as possible, the maxim that 'code that is read together should be written together'.

Everything in xf - models, params, results - is a tuple (or ndarray), which means:

- everythings plays nicely with JAX's auto grad functionality.

- xf models can be manipulated just like any other data structure.

This last point, in particular, allows us to:

- re-use fully- / partially- / un- trained components / pathways of one model as part of another.

- semi-dynamically filter out certain execution paths (say, for training vs scoring vs testing), without impinging on JAX's static shapes constraint (see [here](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.UnexpectedTracerError))

- semantically diff two related models (forthcoming).

As mentioned above, this is still a very much work-in-progress project, that I'm in the process of refactoring out - rewriting large chunks on the way - from our main code base at [Haven Cove](https://havencove.com/).

The unit tests are likely the best place to start for an idea of how the project works in pactise.

Note, the package for now also includes some other convenience utilities from my day job, until I can find a more appropriate long term home for them.

## License

`xfactors` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
