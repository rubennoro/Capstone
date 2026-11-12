"""`python -m deploy` is the `deploy` command."""
from __future__ import annotations

import sys

from deploy.cli import main

if __name__ == "__main__":
    sys.exit(main())
