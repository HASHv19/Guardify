"""
Legacy prediction entrypoint.
"""

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from guardify.predict import main


if __name__ == "__main__":
    main()
