"""
Backward compatibility stub for visualizations module.

.. deprecated::
    This module has been moved to `deprecated/visualizations.py`.
    Use `from visualization import ...` instead.
"""

import warnings
warnings.warn(
    "The visualizations module is deprecated and has been moved to deprecated/. "
    "Use `from visualization import ...` for all plotting functions.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the deprecated module
from deprecated.visualizations import *
