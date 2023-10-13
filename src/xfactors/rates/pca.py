
from __future__ import annotations

from typing import TYPE_CHECKING

import abc
import typing
import dataclasses
import collections

import operator
import itertools
import functools

import datetime

import numpy
import pandas

import jax
import jax.numpy

import xtuples as xt
import xjd

from ... import utils
from ... import visuals
from ... import fin
from ... import bt
from ... import data
from ... import xfactors

# --------------------------------------------------------


# --------------------------------------------------------