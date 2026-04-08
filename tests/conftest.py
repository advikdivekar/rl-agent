import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `models`, `server`, etc. are
# importable regardless of how pytest is invoked (e.g. from the project root,
# from inside tests/, or via a CI runner that doesn't install the package).
# Using insert(0, ...) puts the project root ahead of any installed packages so
# the local source is always preferred over a stale installed version.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
