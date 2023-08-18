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

xfactors (name to be reconsidered) is a (work in progress) library for practical machine learning research in finance, built on [JAX](https://jax.readthedocs.io/en/latest/index.html) (plus some other convenience utilities from my day job, until I can find a more appropriate long term home for them).

xf (xfactors) is designed to make it easier to:

- define machine learning pipelines out of modular, re-usable components.
- obeying, as much as possible, the maxim that 'code that is read together should be written together'.

xf also helps to side-step a couple of JAX limitations, namely:

- making it easier to specify conditional execution without impinging on the static shapes constraint (see [here](https://jax.readthedocs.io/en/latest/index.html))

- making it easier to deal with irregularly shaped data structures (for instance, rolling windows of varying sizes).

## License

`xfactors` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
