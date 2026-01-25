"""
Deprecated modules for backward compatibility.

This package contains modules that have been superseded by:
- `visualization/` - All matplotlib plotting code
- `core/` - All dataclasses and simulation logic

These modules are maintained for backward compatibility but may be removed
in a future release.
"""

# Re-export modules for direct import
from . import visualizations
from . import retirement_simulation
