import sys


def test_missing_numba_import():
    """Test the fallback mechanism when numba is not available."""
    # Store original modules to restore later
    orig_modules = dict(sys.modules)

    # Remove numba if it exists
    if "numba" in sys.modules:
        del sys.modules["numba"]

    # Add a module finder that raises ImportError for numba
    class NumbaBlocker(object):
        def find_spec(self, fullname, path, target=None):
            if fullname == "numba":
                raise ImportError("Simulated ImportError for numba")
            return None

    # Add our blocker to sys.meta_path
    sys.meta_path.insert(0, NumbaBlocker())

    try:
        # Force reload of massphoto to trigger the import code
        if "massphoto" in sys.modules:
            del sys.modules["massphoto"]

        # Also remove any massphoto submodules
        for name in list(sys.modules.keys()):
            if name.startswith("massphoto."):
                del sys.modules[name]

        # Now import massphoto and check the fallback mechanism
        from massphoto import utils

        # Verify fallback decorator behavior
        assert utils.HAS_NUMBA is False

        # Test the no-op decorator fallback
        # When Numba is unavailable the decorator should be a no-op so the function behaves as normal.

        # 1) No parentheses usage
        @utils.numba_njit
        def test_no_paren(x):
            return x * 2

        assert test_no_paren(5) == 10
        assert callable(test_no_paren)

        # 2) With empty parentheses
        @utils.numba_njit()
        def test_empty_paren(x):
            return x * 3

        assert test_empty_paren(4) == 12
        assert callable(test_empty_paren)

        # 3) With signature
        @utils.numba_njit("(int64,int64,int64,int64)")
        def test_with_signature(x):
            return x * 4

        assert test_with_signature(3) == 12
        assert callable(test_with_signature)

        # 4) With signature and kwargs
        @utils.numba_njit("(int64)", cache=False)
        def test_with_sig_kwargs(x):
            return x + 1

        assert test_with_sig_kwargs(5) == 6
        assert callable(test_with_sig_kwargs)

    finally:
        # Restore the original modules
        sys.meta_path.pop(0)  # Remove our blocker
        sys.modules.clear()
        sys.modules.update(orig_modules)
