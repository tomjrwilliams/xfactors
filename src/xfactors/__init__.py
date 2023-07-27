# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import sys

XTUPLES = "C:/hc/xtuples/src/xtuples"
HC = "C:/hc/hc-core/src"

if XTUPLES not in sys.path:
    sys.path.append(XTUPLES)

if HC not in sys.path:
    sys.path.append(HC)