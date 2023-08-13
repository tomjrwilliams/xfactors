# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import sys

XTUPLES = "C:/xtuples/src/xtuples"
XFACTORS = "./src"

if XTUPLES not in sys.path:
    sys.path.append(XTUPLES)

if XFACTORS not in sys.path:
    sys.path.append(XFACTORS)