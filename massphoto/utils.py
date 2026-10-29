"""
Shared definitions used across modules.
Defines:
- `defaults`: Module-wide default parameters.
- `HAS_NUMBA`: Whether Numba JIT is available.
- `numba_njit`: Numba JIT decorator or no-op fallback.
- `NativeMovie`: Type alias for native movie numpy arrays.
- `RatiometricMovie`: Type alias for ratiometric movie numpy arrays.
- `EventList`: Type alias for event list pandas DataFrames.
"""

import logging
from typing import Any, NewType

import numpy as np
import pandas as pd

from .defaults_store import DefaultsStore

logger = logging.getLogger(__name__)

# Module-wide default parameters.
defaults: DefaultsStore = DefaultsStore()

# Optional numba JIT acceleration; falls back to a no-op decorator if unavailable.
try:
    import numba as _numba

    HAS_NUMBA = True
    numba_njit = _numba.njit
except ImportError:
    HAS_NUMBA = False

    def numba_njit(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        """No-op fallback for numba.njit.

        Supports the common usages:
          @numba_njit
          @numba_njit()
          @numba_njit(signature, cache=False)

        The decorator returns the original function unchanged so behavior is identical
        whether Numba is available or not.
        """
        # direct usage without parentheses: @numba_njit
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        # usage with parentheses: @numba_njit(...) -> return a decorator that returns func
        def decorator(func):
            return func

        return decorator

    logger.warning("Numba not available; using no-op JIT decorator. Performance may be reduced.")
# Type aliases to infer inputs in parallel processing functions.
NativeMovie = NewType("NativeMovie", np.ndarray)
RatiometricMovie = NewType("RatiometricMovie", np.ndarray)
EventList = NewType("EventList", pd.DataFrame)
