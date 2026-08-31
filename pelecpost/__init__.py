"""Recipe-driven PeleC post-processing."""

import os
from pathlib import Path
import tempfile


# Matplotlib otherwise tries a home-directory cache that is commonly read-only
# on compute nodes. Respect an explicit server setting and use portable scratch
# only when none was supplied.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pelecpost-matplotlib")
)

__version__ = "0.1.0"
