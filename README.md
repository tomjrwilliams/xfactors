# xfactors

[![PyPI - Version](https://img.shields.io/pypi/v/xfactors.svg)](https://pypi.org/project/xfactors)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xfactors.svg)](https://pypi.org/project/xfactors)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install xfactors
```

## Overview

xfactors (name to be reconsidered) is a work-in-progress library for practical machine learning research in finance, built on [JAX](https://jax.readthedocs.io/en/latest/index.html) (plus some other convenience utilities from my day job, until I can find a more appropriate long term home for them).

xf (xfactors) is designed to make it easier to:

- define machine learning pipelines out of modular, re-usable components.
- obeying, as much as possible, the maxim that 'code that is read together should be written together'.

xf does this by providing:

- base classes / interfaces with which one can define machine learning pipelines as directed acyclic computation graphs of resuable components

- an apparatus for specifying how data, parameters, and intermediate results should flow through each such pipeline

Everything in xf - models, params, results - is a tuple, which means:

- everythings plays nicely with JAX's auto grad functionality

- xf models can be manipulated just like any other data structure

This means, for instance, it's straightforward to:

- semi-dynamically filter out certain execution paths (say, for training vs scoring vs testing).

- re-use fully / partially trained components of one model, as part of another.

- semantically diff two related models (forthcoming).

As such, xf helps to side-step a couple of JAX limitations, namely:

- making it easier to specify conditional execution without impinging on JAX's static shapes constraint (see [here](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.UnexpectedTracerError))

- making it easier to deal with irregularly shaped data structures (for instance, rolling windows of varying sizes).

## License

`xfactors` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
