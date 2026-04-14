"""
CLI entry point shim — delegates to main.main() unchanged.
"""

import sys
import os

# Ensure the project root is on the path so `main.py` and `src/` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
