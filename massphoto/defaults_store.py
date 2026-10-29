"""
Global defaults registry: Shared configuration across the package.
"""

import contextvars
import functools
import inspect
import logging
import os
import sys
from collections.abc import Callable, Collection, Mapping
from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata as importlib_metadata
from threading import RLock
from typing import Any, Generic, ParamSpec, TypeVar, cast

import yaml

logger = logging.getLogger(__name__)

# Generic typing helpers to preserve editor/type-checker view of decorated callables
P = ParamSpec("P")
R = TypeVar("R")

_ctx_state: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("_defaults_ctx_state", default=None)


def _get_runtime_metadata() -> dict[str, Any]:
    try:
        mp_version = importlib_metadata.version("massphoto")
    except importlib_metadata.PackageNotFoundError:
        mp_version = "unknown"
    except Exception:
        mp_version = "unknown"

    return {
        "package_version": mp_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


class DefaultsStore:
    """
    Global defaults registry

    Features:
      - Centralised dictionary store
      - Import-time default registration via decorators
      - Run-time injection of configuration into functions
      - Picklable wrappers for Dask/Multiprocessing workers
      - YAML import/export
      - Context manager for temporary overrides

    Note:
      - All public methods return shallow copies unless ``copy=True`` is specified.
      - Thread-safe for concurrent access.
    """

    _store: dict[str, Any] = {
        "mass_calibration": None,  # example values: (-4.6, 3.0e4) [=(intercept, slope)]: linear contrast to mass in kDa
    }

    _lock: RLock = RLock()

    # ---------------- basic API ----------------
    @classmethod
    def get_state(cls, *, copy: bool = False) -> dict[str, Any]:
        """Return the store."""
        if (ctx := _ctx_state.get()) is not None:
            state = deepcopy(ctx) if copy else dict(ctx)
        else:
            with cls._lock:
                state = deepcopy(cls._store) if copy else dict(cls._store)
        state["__meta__"] = _get_runtime_metadata()
        return state

    @classmethod
    def set_state(cls, state: Mapping[str, Any], *, copy: bool = False) -> None:
        """Replace the store."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        new_state = cls._prepare_state(state, copy=copy)
        if (ctx := _ctx_state.get()) is not None:
            ctx.clear()
            ctx.update(new_state)
            return
        with cls._lock:
            cls._store = new_state

    @staticmethod
    def _prepare_state(state: Mapping[str, Any], *, copy: bool = False) -> dict[str, Any]:
        """Validate and normalize state before applying it."""
        new_state = deepcopy(dict(state)) if copy else dict(state)
        if "__meta__" in new_state:
            meta = _get_runtime_metadata()
            state_version = new_state["__meta__"].get("package_version", "unknown")
            if state_version != meta["package_version"]:
                logger.warning(
                    "Applied defaults state version (%s) does not match current package version (%s)",
                    state_version,
                    meta["package_version"],
                )
            new_state.pop("__meta__", None)
        return new_state

    @classmethod
    def get(cls, key: str, default: Any = None, *, copy: bool = True) -> Any:
        """Get a key from the store."""
        if (ctx := _ctx_state.get()) is not None:
            value = ctx.get(key, default)
            return deepcopy(value) if copy else value
        with cls._lock:
            value = cls._store.get(key, default)
            return deepcopy(value) if copy else value

    @classmethod
    def set(cls, key: str, value: Any, *, copy: bool = False) -> None:
        """Set a key in the store."""
        if (ctx := _ctx_state.get()) is not None:
            ctx[key] = deepcopy(value) if copy else value
            return
        with cls._lock:
            cls._store[key] = deepcopy(value) if copy else value

    @classmethod
    def update(cls, mapping: Mapping[str, Any], *, copy: bool = False) -> None:
        """Update multiple keys."""
        if (ctx := _ctx_state.get()) is not None:
            for k, v in mapping.items():
                ctx[k] = deepcopy(v) if copy else v
            return
        with cls._lock:
            for k, v in mapping.items():
                cls._store[k] = deepcopy(v) if copy else v

    @classmethod
    def pop(cls, key: str, *args: Any) -> Any:
        """Remove ``key`` from the store and return its previous value.

        Behaves like :py:meth:`dict.pop`:

        - If a default is supplied (``pop('x', default)``) it is returned when
          the key is missing.
        - If no default is supplied (``pop('x')``) a ``KeyError`` is raised
          when the key is missing.

        Example::

            # raises KeyError if 'x' is absent
            massphoto.defaults.pop('x')

            # returns None instead of raising
            massphoto.defaults.pop('x', None)
        """
        if (ctx := _ctx_state.get()) is not None:
            return ctx.pop(key, *args)
        with cls._lock:
            return cls._store.pop(key, *args)

    # ---------------- attribute-style access ----------------
    def __getattr__(self, key: str) -> Any:
        return self.__class__.get(key, copy=False)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("__") and key.endswith("__"):
            # don't interfere with special attributes
            return super().__setattr__(key, value)
        self.__class__.set(key, value)

    # ---------------- I/O ----------------
    @classmethod
    def save(cls, path: str | os.PathLike, *, unsafe: bool = False) -> None:
        """Save store to YAML."""
        store = cls.get_state(copy=False)
        with open(path, "w", encoding="utf-8") as f:
            if unsafe:
                yaml.dump(store, f)
            else:
                yaml.safe_dump(store, f)

    @staticmethod
    def _load_yaml(path: str | os.PathLike, unsafe: bool = False) -> dict[str, Any]:
        """Internal: Load YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader) if unsafe else yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Defaults file must contain a mapping")
        return data

    @classmethod
    def load(cls, path: str | os.PathLike, *, unsafe: bool = False, copy: bool = False) -> None:
        """Load YAML into store."""
        data = cls._load_yaml(path, unsafe=unsafe)
        cls.set_state(data, copy=copy)

    # ---------------- import-time signature sync ----------------
    @classmethod
    def _sync_signature_defaults(
        cls,
        func: Callable[..., Any],
        *,
        prefix: str | None = None,
        overwrite: bool = False,
        exclude: Collection[str] | None = None,
        promote: Collection[str] | None = None,
    ) -> None:
        """
        Internal: Register function defaults into the store.
        Keys registered: "Func.param" and (optional) "prefix.param".
        """
        logger.debug("Syncing defaults for %s", func.__name__)
        if _ctx_state.get() is not None:
            # this might lead to confusing behavior, so warn
            logger.warning(
                "DefaultsStore._sync_signature_defaults called within context; modifications will be local to the context"
            )
        sig = inspect.signature(func)
        store = cls.get_state(copy=False)
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                continue
            if exclude and name in exclude:
                continue

            if promote and name in promote:
                param_key = name
            elif prefix is not None:
                param_key = f"{prefix}.{name}"
            else:
                param_key = f"{func.__name__}.{name}"

            if param_key in store and not overwrite:
                logger.warning(
                    "Skipping registration of default for %s.%s: key '%s' exists and overwrite=False",
                    func.__name__,
                    name,
                    param_key,
                )
                continue
            cls.set(param_key, param.default, copy=True)

    @classmethod
    def sync_defaults(
        cls,
        *,
        prefix: str | None = None,
        overwrite: bool = False,
        exclude: Collection[str] | None = None,
        promote: Collection[str] | None = None,
    ):
        """
        Decorator: Registers signature defaults at import time.
        Does NOT modify runtime behavior.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cls._sync_signature_defaults(
                func,
                prefix=prefix,
                overwrite=overwrite,
                exclude=exclude,
                promote=promote,
            )
            return func

        return decorator

    @classmethod
    def apply_defaults(
        cls,
        *,
        prefix: str | None = None,
    ):
        """
        Decorator: Injects store values at runtime.
        Use :py:meth:`functools.wraps`: to preserve function metadata (introspection, Sphinx, type-checkers).
        Store _injector on the wrapper for pickling support.
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            # Create an injector and return a real function wrapper so Sphinx/inspect
            # and type-checkers see a normal function.
            injector = FunctionInjector(func, prefix=prefix)

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return injector(*args, **kwargs)

            # Keep a reference to the injector for tests and wrap_callable.
            wrapper._injector = injector  # pylint: disable=protected-access # type: ignore[reportAttributeAccessIssue]
            wrapper.__signature__ = injector.__signature__  # type: ignore[reportAttributeAccessIssue]
            return cast(Callable[P, R], wrapper)

        return decorator

    @classmethod
    def sync_apply_defaults(
        cls,
        *,
        prefix: str | None = None,
        overwrite: bool = False,
        exclude: Collection[str] | None = None,
        promote: Collection[str] | None = None,
    ):
        """
        Decorator: Registers defaults AND injects store values at runtime.
        Combines `sync_defaults` and `apply_defaults`.
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            cls._sync_signature_defaults(
                func,
                prefix=prefix,
                overwrite=overwrite,
                exclude=exclude,
                promote=promote,
            )
            return cls.apply_defaults(
                prefix=prefix,
            )(func)

        return decorator

    # ---------------- worker wrapper ----------------
    @classmethod
    def wrap_callable(
        cls,
        func: Callable[P, R],
        state: Mapping[str, Any] | None = None,
        copy: bool = True,
    ) -> Callable[P, R]:
        """
        Prepare a callable for a worker process (Dask/Multiprocessing).
        Ensures the worker initializes the store with 'state' (or current state)
        before execution.
        """
        if not callable(func):
            logger.debug("wrap_callable: func is not callable, returning unwrapped")
            return func

        # unwrap if already wrapped
        if getattr(func, "_injector", None) is not None:
            func = func._injector
        if isinstance(func, _WithDefaultsState):
            func = func.func

        snapshot = dict(state) if state is not None else cls.get_state()
        return cast(Callable[P, R], _WithDefaultsState(func, deepcopy(snapshot) if copy else snapshot))

    # ---------------- context manager ----------------
    @classmethod
    @contextmanager
    def context(
        cls,
        overrides: Mapping[str, Any] | None = None,
        *,
        copy: bool = True,
        load: os.PathLike | None = None,
        unsafe: bool = False,
    ):
        """Temporarily override store values within a context block.

        Parameters
        ----------
        overrides : Mapping[str, Any] | None, optional
            Mapping of keys and values to override in the store.
        copy : bool, default=True
            If True, overrides and state snapshots are deep-copied into the context.
        load : os.PathLike | None, optional
            Optional path to a YAML file to load into the context.
        unsafe : bool, default=False
            If True, use PyYAML's unsafe loader when loading from YAML.

        Yields
        ------
        type[DefaultsStore]
            The DefaultsStore class with temporary overrides active.

        Examples
        --------
        >>> with mp.defaults.context({"fit.threshold": 0.005}):
        ...     events = mp.fit("movie.mp")

        >>> mp.defaults.set("navg", 15)
        ... print(mp.defaults.get("navg"))          # 15
        ... with mp.defaults.context({"navg": 5}):
        ...     print(mp.defaults.get("navg"))      # 5

        ... print(mp.defaults.get("navg"))          # 15
        """
        # Create a new context state
        if load is not None:
            state = cls._load_yaml(load, unsafe=unsafe)
        else:
            state = cls.get_state(copy=True)
        if overrides:
            state |= dict(overrides)
        token = _ctx_state.set(deepcopy(state) if copy else state)
        try:
            yield cls
        finally:
            _ctx_state.reset(token)


# ---------------------- Helper Classes ----------------------


class FunctionInjector(Generic[P, R]):
    """
    Callable wrapper that injects configuration values from DefaultsStore
    at call time.

    The wrapper behaves like the wrapped function for introspection
    (signature, docstring, annotations) so that inspect and Sphinx
    autodoc treat it as a normal function.
    """

    def __init__(self, func: Callable[P, R], *, prefix: str | None = None) -> None:
        # Unwrap nested injectors if present
        inner = getattr(func, "func", None)
        self.func: Callable[P, R] = cast(Callable[P, R], inner) if callable(inner) else func
        self.__signature__: inspect.Signature = inspect.signature(self.func)
        self.prefix = prefix
        self._expose_metadata()

    def _expose_metadata(self) -> None:
        """
        Attach function-like attributes directly to this instance so that
        inspect, help(), and Sphinx autodoc see a regular function.
        """
        self.__wrapped__ = self.func
        self.__name__ = getattr(self.func, "__name__", type(self.func).__name__)
        self.__doc__ = getattr(self.func, "__doc__", None)
        self.__module__ = getattr(self.func, "__module__", type(self.func).__module__)
        self.__qualname__ = getattr(self.func, "__qualname__", self.__name__)
        self.__annotations__ = dict(getattr(self.func, "__annotations__", {}))

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes to the wrapped function
        return getattr(self.func, name)

    def __getstate__(self):
        # Pickle only minimal state; the signature is recomputed on restore
        return {"func": self.func, "prefix": self.prefix}

    def __setstate__(self, state):
        self.func = state["func"]
        self.prefix = state["prefix"]
        self.__signature__ = inspect.signature(self.func)
        self._expose_metadata()

    def get_args(self, *args: P.args, **kwargs: P.kwargs) -> inspect.BoundArguments:
        """
        Get bound arguments with defaults injected from the store.
        See :py:meth:`__call__` for the injection order.

        Parameters
        ----------
        *args, **kwargs
            Arguments to bind.

        Returns
        -------
        inspect.BoundArguments
            Bound arguments with injected defaults.
        """

        # Bind provided arguments to parameter names
        sig = self.__signature__
        bound = sig.bind_partial(*args, **kwargs)
        arguments = bound.arguments
        store = DefaultsStore.get_state(copy=False)  # shared defaults registry

        for name, param in sig.parameters.items():
            # Skip parameters already provided by the caller
            if name in arguments:
                continue

            # Never inject into *args or **kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            val = inspect.Parameter.empty

            func_key = f"{self.func.__name__}.{name}"
            pref_key = f"{self.prefix}.{name}" if self.prefix else None

            if func_key in store:
                val = store[func_key]
            elif pref_key and pref_key in store:
                val = store[pref_key]
            elif name in store:
                val = store[name]
            elif param.default is not inspect.Parameter.empty:
                val = param.default

            if val is not inspect.Parameter.empty:
                arguments[name] = val

        return bound

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Call the wrapped function with defaults injected.

        Argument resolution order (highest precedence first):

        1. Explicit arguments supplied by the caller.
        2. Function-specific defaults:
           ``<function_name>.<parameter>``
        3. Prefix defaults (if a prefix is configured):
           ``<prefix>.<parameter>``
        4. Global defaults:
           ``<parameter>``
        5. The original default value from the function signature.

        ``*args`` and ``**kwargs`` parameters are never injected.
        """
        logger.debug("Injecting defaults for %s", self.func.__name__)
        bound = self.get_args(*args, **kwargs)
        return cast(R, self.func(*bound.args, **bound.kwargs))


class _WithDefaultsState:
    """
    Picklable wrapper that initialises DefaultsStore on the worker process
    on first call.
    """

    __slots__ = ("func", "state", "_pid")

    def __init__(self, func: Callable[..., Any], state: dict[str, Any]):
        self.func = func
        self.state = state
        self._pid: int | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with DefaultsStore.context(self.state, copy=True):
            return self.func(*args, **kwargs)
