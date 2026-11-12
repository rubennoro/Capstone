# from rnoro/data-work:data-processing/dataproc/paths.py @ a1ddbbc
# changes: only the NAME_RE line (and its imports) kept
from __future__ import annotations

import re

#: Allowed task, folder and rig names.
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
