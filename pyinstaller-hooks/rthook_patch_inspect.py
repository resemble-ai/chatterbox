"""
Runtime hook to patch inspect module for PyInstaller compatibility.

The transformers library uses inspect.getsource() in decorators, which fails
in PyInstaller bundles since source code is not available. This hook patches
inspect to gracefully handle these cases.
"""

import sys
import inspect

# Store original functions
_original_getsource = inspect.getsource
_original_getsourcelines = inspect.getsourcelines
_original_findsource = inspect.findsource

def patched_getsource(object):
    """Return minimal valid source if source cannot be found instead of raising OSError."""
    try:
        return _original_getsource(object)
    except (OSError, TypeError):
        # Return minimal valid Python source when source is not available (PyInstaller bundle)
        # This prevents IndexError in transformers code that expects at least one line
        return "pass\n"

def patched_getsourcelines(object):
    """Return minimal valid source lines if source lines cannot be found instead of raising OSError."""
    try:
        return _original_getsourcelines(object)
    except (OSError, TypeError):
        # Return minimal valid source (one line) and line number 0 when source is not available
        # This prevents IndexError in transformers doc.py that expects at least one line
        return (["pass\n"], 0)

def patched_findsource(object):
    """Return minimal valid source if source cannot be found instead of raising OSError."""
    try:
        return _original_findsource(object)
    except (OSError, TypeError):
        # Return minimal valid source (one line) and line number 0 when source is not available
        return (["pass\n"], 0)

# Apply patches
inspect.getsource = patched_getsource
inspect.getsourcelines = patched_getsourcelines
inspect.findsource = patched_findsource

print("[PyInstaller Runtime Hook] Patched inspect module for frozen environment")
