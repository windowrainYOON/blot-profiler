"""Entry point used by PyInstaller and for running from source: python run_blot_profiler.py"""

import sys

from blot_profiler.gui import main

if __name__ == "__main__":
    sys.exit(main())
