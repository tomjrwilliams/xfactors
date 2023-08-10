# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import sys

XTUPLES = "C:/hc/xtuples/src/xtuples"
XTENORS = "C:/hc/xtenors/src/xtenors"

if XTUPLES not in sys.path:
    sys.path.append(XTUPLES)

if XTENORS not in sys.path:
    sys.path.append(XTENORS)

from . import bts
from . import nodes
from . import utils
from . import visuals

from .xfactors import *
