# SPDX-FileCopyrightText: 2023-present Tom Williams <tomjrw@gmail.com>
#
# SPDX-License-Identifier: MIT

import os
import sys

sys.path.append("./__local__")
sys.path.append("./src")

sys.path.append(os.environ["xtuples"])
sys.path.append(os.environ["xsm"])
sys.path.append(os.environ["xtenors"])
sys.path.append(os.environ["xrates"])