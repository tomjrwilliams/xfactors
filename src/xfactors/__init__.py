# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import sys
import pathlib

if pathlib.Path(os.getcwd()).parts[-1] == "xfactors":
    sys.path.append("./__local__")

    import PATHS

    if PATHS.XTUPLES not in sys.path:
        sys.path.append(PATHS.XTUPLES)

    if PATHS.XTENORS not in sys.path:
        sys.path.append(PATHS.XTENORS)

from . import utils
from . import visuals
from . import nodes
from . import bt

from .nodes import *
from .xfactors import *
