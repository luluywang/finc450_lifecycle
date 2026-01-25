"""
Backward compatibility stub for retirement_simulation module.

.. deprecated::
    This module has been moved to `deprecated/retirement_simulation.py`.
    Use `from core import ...` for dataclasses and simulation functions.
"""

import warnings
warnings.warn(
    "The retirement_simulation module is deprecated and has been moved to deprecated/. "
    "Use `from core import ...` for dataclasses and simulation functions.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the deprecated module
from deprecated.retirement_simulation import *
