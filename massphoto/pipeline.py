from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from inspect import signature
from typing import NamedTuple, get_type_hints

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp
from tqdm.dask import TqdmCallback

from .defaults_store import FunctionInjector
from .movie_drift_correction import correct_drift, get_drift_metrics, get_shift_rfft
from .movie_process import calc_frame_sharpness, calc_ratiometric
from .read import read_movie
from .utils import EventList, NativeMovie, RatiometricMovie, defaults

logger = logging.getLogger(__name__)

_ANALYSIS_RECORD_SCHEMA_VERSION = 2


class _StepSpec(NamedTuple):
    """Internal, normalized pipeline step."""

    func: Callable
    kwargs: dict
    arg_sources: list[str]
    overlap: int


def _infer_callable_arg_sources(func: Callable) -> list[str]:
    """
    Infer which inputs (native/ratiometric/events) a callable expects based on its
    type annotations, preserving parameter order.
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    sig = signature(func)
    arg_sources: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        if annotation == NativeMovie:
            arg_sources.append("native")
        elif annotation == RatiometricMovie:
            arg_sources.append("ratiometric")
        elif annotation in (EventList, pd.DataFrame):
            arg_sources.append("events")
    return arg_sources


def _infer_chunk_overlap(
    *,
    step_specs: Sequence[_StepSpec],
    ratiometric_args: dict | None,
    needs_ratiometric: bool,
    chunk_overlap: int,
) -> int:
    if chunk_overlap != -1:
        return int(chunk_overlap)

    overlap = 0
    if needs_ratiometric and ratiometric_args is not None:
        navg = int(ratiometric_args.get("navg", 0) or 0)
        ngap = int(ratiometric_args.get("ngap", 1) or 1)
        overlap = max(overlap, navg + ngap)

    overlap = max(overlap, *(spec.overlap for spec in step_specs))

    return int(overlap)


def _effective_step_kwargs(func: Callable, kwargs: dict) -> dict:
    """Return step kwargs after applying function signature defaults."""
    bound = FunctionInjector(func).get_args(**kwargs)
    effective_kwargs = dict(bound.arguments)
    for name, param in signature(func).parameters.items():
        if param.kind == param.VAR_KEYWORD and isinstance(effective_kwargs.get(name), dict):
            effective_kwargs.update(effective_kwargs.pop(name))
    return effective_kwargs


def _infer_step_overlap(func: Callable, kwargs: dict) -> int:
    """Infer the temporal overlap required by one processing step."""
    effective_kwargs = _effective_step_kwargs(func, kwargs)
    overlap = 0

    for key in ("min_distance_temp", "threshold_distance_temp"):
        overlap = max(overlap, int(effective_kwargs.get(key, 0) or 0))

    if "correlation_step_args" in effective_kwargs:
        from .filter.temporal_ncc import step_correlation

        step_args = effective_kwargs["correlation_step_args"]
        step_kwargs = step_args if isinstance(step_args, dict) else {}
        step_bound = FunctionInjector(step_correlation).get_args(**step_kwargs)
        overlap = max(overlap, int(step_bound.arguments.get("navg", 0) or 0))

    return int(overlap)


def _compute_chunk_sizes(movie_length: int, chunk_size: int, overlap: int = 0) -> tuple[int, ...]:
    """
    Compute chunk sizes for processing a movie in parallel, given the total length, desired chunk size, and overlap between chunks.
    Optimises for equal chunk size and ensures that all chunks are larger than the overlap (if movie_length>chunk_size).
    """
    if movie_length <= max(chunk_size, overlap):
        return (movie_length,)

    n_chunks = int(np.ceil(movie_length / chunk_size))
    max_chunks = movie_length // max(1, overlap)
    n_chunks = min(n_chunks, max_chunks)

    edges = np.linspace(0, movie_length, n_chunks + 1).round().astype(int)
    return tuple(np.diff(edges))


def _apply_postprocess(
    df: pd.DataFrame,
    postprocess: Sequence[Callable[[pd.DataFrame], pd.DataFrame]],
) -> pd.DataFrame:
    for func in postprocess:
        df = func(df)
    return df


def _normalize_events(events: pd.DataFrame | np.ndarray | list | None) -> pd.DataFrame | None:
    if events is None:
        return None

    if isinstance(events, pd.DataFrame):
        events_df = events.copy()

        if events_df.empty:
            return pd.DataFrame(columns=["frame", "y", "x"])

        required_columns = {"frame", "y", "x"}
        if not required_columns.issubset(events_df.columns):
            raise ValueError("events DataFrame must contain 'frame', 'y', and 'x' columns")
        return events_df

    events_arr = np.asarray(events)
    if events_arr.size == 0:
        return pd.DataFrame(columns=["frame", "y", "x"])

    if events_arr.ndim != 2 or events_arr.shape[1] != 3:
        raise ValueError("events array/list must be 2D with shape (n, 3)")

    return pd.DataFrame(events_arr, columns=["frame", "y", "x"])


def _chunk_process_pipeline(
    movie_native: np.ndarray | None,
    movie_ratiometric: np.ndarray | None,
    *,
    step_specs: Sequence[_StepSpec],
    chunk_length: int | None = None,
    chunk_offset: int = 0,
    chunk_overlap: int = 0,
    seed_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    events: pd.DataFrame | None = seed_events.copy() if seed_events is not None else None

    for spec in step_specs:
        args = []
        for src in spec.arg_sources:
            if src == "native":
                if movie_native is None:
                    raise ValueError("Step requires native movie but none was provided")
                args.append(movie_native)
            elif src == "ratiometric":
                if movie_ratiometric is None:
                    raise ValueError("Step requires ratiometric movie but none was provided")
                args.append(movie_ratiometric)
            elif src == "events":
                if events is None:
                    raise ValueError("Step requires EventList but none is available yet")
                args.append(events)
            else:
                raise ValueError(f"Unknown arg source: {src}")

        events = spec.func(*args, **spec.kwargs)

    if events is None:
        return pd.DataFrame(columns=["frame", "y", "x"])
    if not isinstance(events, pd.DataFrame):
        raise ValueError(f"Step output must be a pandas DataFrame, got {type(events).__name__}")
    if events.empty:
        return events

    if "frame" not in events.columns:
        raise ValueError("EventList must have a 'frame' column")

    if chunk_length is not None:
        movie_len = int(chunk_length) + (2 * int(chunk_overlap))
    elif movie_native is not None:
        movie_len = movie_native.shape[0]
    elif movie_ratiometric is not None:
        movie_len = movie_ratiometric.shape[0]
    else:
        raise ValueError("At least one movie input or chunk_length must be provided")
    ind = np.bitwise_and(chunk_overlap <= events["frame"], events["frame"] < movie_len - chunk_overlap)
    if chunk_offset or chunk_overlap:
        events["frame"] += chunk_offset - chunk_overlap
    events_out = events.loc[ind]
    return events_out


def _empty_movie_like(movie_like: da.Array | np.ndarray) -> np.ndarray:
    shape = tuple(movie_like.shape)
    if len(shape) < 1:
        raise ValueError("movie must have at least one dimension")
    return np.empty((0, *(int(s) for s in shape[1:])), dtype=getattr(movie_like, "dtype", float))


def _capture_origin_kwargs(
    func: Callable,
    local_vars: Mapping[str, object],
    *,
    exclude: set[str] | None = None,
) -> tuple[str, dict]:
    """Capture origin name and call kwargs for ``func`` from local variables."""
    exclude = exclude or set()
    origin = func.__name__
    func_signature = signature(func)
    captured: dict = {}
    for name, param in func_signature.parameters.items():
        if name in exclude or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in local_vars:
            captured[name] = deepcopy(local_vars[name])
    return origin, captured


def _apply_analysis_record(
    result: pd.DataFrame | tuple[pd.DataFrame, np.ndarray],
    *,
    attach_record: bool,
    func: Callable,
    local_vars: Mapping[str, object],
    exclude: set[str] | None = None,
    movie_info: dict | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    if not attach_record:
        return result

    df = result[0] if isinstance(result, tuple) else result
    if not isinstance(df, pd.DataFrame):
        return result

    record = df.attrs.get("analysis_record", {})
    if not isinstance(record, dict):
        record = {}

    if "defaults_state" not in record:
        record["defaults_state"] = defaults.get_state(copy=True)

    origin, origin_kwargs = _capture_origin_kwargs(func, local_vars, exclude=exclude)

    record["schema_version"] = _ANALYSIS_RECORD_SCHEMA_VERSION
    record["origin"] = origin
    record["origin_kwargs"] = origin_kwargs
    if movie_info is not None:
        record["movie_info"] = movie_info
    elif "movie_info" not in record:
        record["movie_info"] = {}

    df.attrs["analysis_record"] = record
    return result


def resolve_record(
    obj: dict | pd.DataFrame,
    *,
    apply_defaults: bool = False,
) -> tuple[dict, Callable]:
    """Resolve an analysis record into defaults overrides and a replay callable.

    Parameters
    ----------
    obj : dict or pd.DataFrame
        Either an ``analysis_record`` dictionary or a DataFrame carrying one in
        ``obj.attrs["analysis_record"]``.
    apply_defaults : bool, default=False
        If True, apply the resolved defaults state globally via
        :data:`massphoto.defaults`.

    Returns
    -------
    tuple[dict, Callable]
        ``(defaults_overrides, replay_callable)`` where:

        - ``defaults_overrides`` is the merged defaults state inferred from the
          record and its origin kwargs.
        - ``replay_callable`` is a defaults-wrapped callable for the recorded
          origin (``process``, ``detect``, or
          ``fit``), preloaded with ``defaults_overrides``.

    Raises
    ------
    TypeError
        If ``obj`` is neither a record dict nor a DataFrame with dict
        ``attrs['analysis_record']``, or if ``origin_kwargs`` is not a dict.
    ValueError
        If required record keys are missing, the schema version is unsupported,
        or the record origin is unknown. The supported schema version is 2.
    """
    record = obj.attrs.get("analysis_record") if isinstance(obj, pd.DataFrame) else obj
    if not isinstance(record, dict):
        raise TypeError("obj must be a record dict or a DataFrame with dict attrs['analysis_record']")

    missing = {"schema_version", "origin", "defaults_state", "origin_kwargs"} - set(record)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"analysis_record is missing required keys: {missing_str}")

    if record["schema_version"] != _ANALYSIS_RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported analysis_record schema_version: {record['schema_version']!r}; "
            f"expected {_ANALYSIS_RECORD_SCHEMA_VERSION}"
        )

    defaults_state = deepcopy(record["defaults_state"])
    origin = str(record["origin"])
    origin_kwargs = deepcopy(record["origin_kwargs"])
    if not isinstance(origin_kwargs, dict):
        raise TypeError("analysis_record['origin_kwargs'] must be a dict")

    defaults_overrides = deepcopy(defaults_state)
    for key, value in origin_kwargs.items():
        defaults_overrides[f"{origin}.{key}"] = value

    if origin == "fit":
        from .fitting import fit

        run_func = fit
    elif origin == "detect":
        from .detection import detect

        run_func = detect
    elif origin == "process":
        run_func = process
    else:
        raise ValueError(f"Unknown record origin: {origin}")

    if apply_defaults:
        logger.info("Applying defaults state from analysis_record for origin '%s'", origin)
        defaults.set_state(defaults_overrides, copy=True)

    replay_callable = defaults.wrap_callable(run_func, state=defaults_overrides, copy=True)
    return defaults_overrides, replay_callable


@defaults.apply_defaults()
def process(
    movie: da.Array | np.ndarray | os.PathLike,
    steps: Sequence[tuple[Callable, dict]],
    *,
    events: pd.DataFrame | np.ndarray | list | None = None,
    movie_template: np.ndarray | None = None,
    movie_normalize_frame_means: bool = False,
    movie_correct_drift: bool = False,
    return_ratiometric: bool = False,
    ratiometric_args: dict | None = None,
    ratiometric_func: Callable | None = calc_ratiometric,
    chunk_overlap: int = -1,
    chunk_size: int = 1000,
    postprocess: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = (),
    scheduler: str | None = None,
    progress_bar: bool = True,
    attach_record: bool = True,
    compute_movie_metrics: bool = False,
):
    """
    Run a chunked, parallel processing pipeline over a movie.

    Each step is a `(callable, kwargs)` pair. Inputs for each callable are inferred
    from type annotations (NativeMovie, RatiometricMovie, EventList/pd.DataFrame).
    Optional precomputed `events` can be provided to seed the pipeline.
    ``movie_template`` is the global native reference image used by movie-level
    preprocessing. When ``movie_normalize_frame_means=True``, each native frame
    is scaled so its mean matches ``movie_template.mean()``. When
    ``movie_correct_drift=True``, the same template is passed to drift
    correction. If needed and not supplied, the template is computed as the mean
    image of the input movie.

    The optional `postprocess` callables are applied to the final (non-chunked)
    DataFrame after compute.

    When ``attach_record=True`` (default), the returned DataFrame stores an
    ``analysis_record`` in ``DataFrame.attrs``. The record captures enough
    provenance to replay the analysis with equivalent defaults context:

    - ``defaults_state``: a copy of :data:`massphoto.defaults` at runtime.
    - ``origin`` and ``origin_kwargs``: the entry-point function name and the
      captured call kwargs.
    - ``movie_info``: input movie shape and dtype.

    For tuple outputs (``return_ratiometric=True``), the record is attached to
    the DataFrame element. Use :func:`resolve_record` to reconstruct a replay
    callable from the stored record.
    """
    with TqdmCallback(desc="process movie chunks", disable=not progress_bar):
        ## load movie if a path is provided
        if isinstance(movie, (str, os.PathLike)):
            movie_path = movie
            logger.info("Loading movie from %s", movie_path)
            movie, _ = read_movie(movie_path)
        if not isinstance(movie, (da.Array, np.ndarray)):
            raise TypeError(f"movie must be a dask array, numpy array, or file path, got {type(movie).__name__}")
        empty_movie = _empty_movie_like(movie)
        movie_shape = tuple(int(s) for s in movie.shape)
        movie_dtype = str(movie.dtype)
        movie_info = {"shape": movie_shape, "dtype": movie_dtype}

        ## validation / parameter computation
        step_specs: list[_StepSpec] = []
        needs_native = False
        needs_ratiometric = False
        for func, kwargs in steps:
            kwargs = kwargs or {}
            arg_sources = _infer_callable_arg_sources(func)
            needs_native |= "native" in arg_sources
            needs_ratiometric |= "ratiometric" in arg_sources
            step_specs.append(
                _StepSpec(
                    func=defaults.wrap_callable(func),
                    kwargs=kwargs,
                    arg_sources=arg_sources,
                    overlap=_infer_step_overlap(func, kwargs),
                )
            )

        needs_native |= compute_movie_metrics
        needs_ratiometric |= return_ratiometric or compute_movie_metrics
        logger.debug(
            "process flags: needs_native=%s needs_ratiometric=%s",
            needs_native,
            needs_ratiometric,
        )

        # Merge user-provided ratiometric_args with defaults from the ratiometric_func signature
        ratiometric_args = ratiometric_args or {}
        ratiometric_dtype = empty_movie.dtype
        if callable(ratiometric_func):
            injector = FunctionInjector(ratiometric_func)
            bound = injector.get_args(**ratiometric_args)
            ratiometric_args = {**bound.kwargs}
            ratiometric_dtype = injector(empty_movie).dtype

        _overlap = _infer_chunk_overlap(
            step_specs=step_specs,
            ratiometric_args=ratiometric_args,
            needs_ratiometric=needs_ratiometric,
            chunk_overlap=chunk_overlap,
        )
        events_df = _normalize_events(events)

        if isinstance(movie, da.Array) and movie.chunks[0] is not None:
            logger.debug("Movie provided as dask array with chunks %s", getattr(movie, "chunks", None))
            chunk_sizes = tuple(int(s) for s in movie.chunks[0])
            movie_dask = movie
        else:
            # calculate "optimal" chunk size
            chunk_sizes = _compute_chunk_sizes(int(movie.shape[0]), chunk_size, _overlap)
            logger.debug("Inferred chunk_overlap=%d, Computed chunk sizes=%s", _overlap, chunk_sizes)
            movie_dask = da.from_array(movie, chunks=(chunk_sizes, -1, -1))

        if _overlap >= min(chunk_sizes):
            logger.warning(
                "Chunk overlap %d is larger than smallest chunk size %d. "
                "This will be slow and may lead to incorrect results at chunk boundaries.",
                _overlap,
                min(chunk_sizes),
            )

        if movie_template is None and (movie_normalize_frame_means or movie_correct_drift):
            logger.info("Computing movie template")
            movie_template = movie_dask.mean(axis=0).compute(scheduler=scheduler)
        elif movie_template is not None:
            movie_template = np.asarray(movie_template)
            expected_template_shape = tuple(int(s) for s in movie_dask.shape[1:])
            if movie_template.shape != expected_template_shape:
                raise ValueError(
                    f"movie_template must have shape {expected_template_shape}, got {movie_template.shape}"
                )

        if movie_normalize_frame_means:
            logger.info("Normalizing movie frame means")
            target_mean = float(np.mean(movie_template))
            frame_means = movie_dask.mean(axis=(1, 2))
            safe_means = da.where(frame_means == 0, 1, frame_means)
            movie_dask = movie_dask * (target_mean / safe_means)[:, None, None]

        drift_yx_full = None
        if movie_correct_drift:
            logger.info("Computing drift trace and applying drift correction")
            template_rfft = sp.fft.rfft2(movie_template)
            drift_dask = movie_dask.map_blocks(
                lambda chunk: get_shift_rfft(chunk, template_rfft=template_rfft),
                chunks=(movie_dask.chunks[0], (2,)),
                drop_axis=(1, 2),
                new_axis=1,
                dtype=np.float32,
            )
            drift_yx_full = drift_dask.compute(scheduler=scheduler)
            movie_dask = da.map_blocks(
                lambda chunk, block_info=None: correct_drift(
                    chunk,
                    drift_yx=(
                        drift_yx_full[block_info[0]["array-location"][0][0] : block_info[0]["array-location"][0][1]]
                        if block_info
                        else None
                    ),
                ),
                movie_dask,
                dtype=movie_dask.dtype,
            )

        if needs_ratiometric:
            if callable(ratiometric_func):
                if "navg" not in ratiometric_args:
                    raise ValueError("navg must be specified in ratiometric_args")
                logger.info(
                    "Computing ratiometric movie (navg=%s, ngap=%s)",
                    ratiometric_args.get("navg"),
                    ratiometric_args.get("ngap", 1),
                )
                movie_r_dask = da.map_overlap(
                    defaults.wrap_callable(ratiometric_func),
                    movie_dask,
                    **ratiometric_args,
                    depth={
                        0: ratiometric_args["navg"] + ratiometric_args.get("ngap", 1),
                        1: 0,
                        2: 0,
                    },
                    boundary={0: "none"},
                    trim=True,
                    allow_rechunk=False,
                    meta=np.empty((0,) * movie_dask.ndim, dtype=ratiometric_dtype),
                )
            else:
                if needs_native:
                    logger.warning("Analysis requires native and ratiometric movies but no ratiometric_func provided.")
                movie_r_dask = movie_dask
        else:
            movie_r_dask = None

        if needs_native:
            movie_native_over = da.overlap.overlap(movie_dask, depth={0: _overlap, 1: 0, 2: 0}, boundary={0: 0})
            native_chunks = list(movie_native_over.to_delayed().ravel())
        else:
            native_chunks = None

        if needs_ratiometric:
            movie_ratio_over = da.overlap.overlap(movie_r_dask, depth={0: _overlap, 1: 0, 2: 0}, boundary={0: 0})
            ratio_chunks = list(movie_ratio_over.to_delayed().ravel())
        else:
            ratio_chunks = None
        offsets = np.cumsum(np.pad(movie_dask.chunks[0], (1, 0)))
        chunk_lengths = movie_dask.chunks[0]

        if events_df is not None and len(events_df):
            chunk_seed_events = []
            for i in range(len(offsets) - 1):
                chunk_start = int(offsets[i]) - _overlap
                chunk_stop = int(offsets[i]) + int(chunk_lengths[i]) + _overlap
                ind = np.bitwise_and(events_df["frame"] >= chunk_start, events_df["frame"] < chunk_stop)
                events_chunk = events_df.loc[ind].copy()
                if len(events_chunk):
                    events_chunk["frame"] -= chunk_start
                chunk_seed_events.append(events_chunk)
        else:
            chunk_seed_events = [None] * (len(offsets) - 1)

        chunk_dfs = [
            dask.delayed(_chunk_process_pipeline)(
                native_chunks[i] if native_chunks is not None else None,
                ratio_chunks[i] if ratio_chunks is not None else None,
                step_specs=step_specs,
                chunk_length=int(chunk_lengths[i]),
                chunk_offset=int(offsets[i]),
                chunk_overlap=_overlap,
                seed_events=chunk_seed_events[i],
            )
            for i in range(len(offsets) - 1)
        ]

        empty_events_meta = _chunk_process_pipeline(
            empty_movie if needs_native else None,
            empty_movie.astype(ratiometric_dtype) if needs_ratiometric else None,
            step_specs=step_specs,
            chunk_length=0,
            seed_events=events_df.iloc[0:0].copy() if events_df is not None else None,
        )

        ## movie metrics computation setup
        metric_tasks: dict[str, da.Array] = {}
        if compute_movie_metrics:
            metric_tasks["intensity"] = movie_dask.mean(axis=(1, 2))
            metric_tasks["sharpness"] = movie_dask.map_blocks(
                calc_frame_sharpness,
                drop_axis=(1, 2),
                chunks=(movie_dask.chunks[0],),
                dtype=np.float32,
            )
            if movie_r_dask is not None:
                metric_tasks["signal"] = movie_r_dask.std(axis=(1, 2))

        ## compute and postprocess in a single pass without dask-expr
        to_compute: list[object] = list(chunk_dfs)
        n_chunks = len(chunk_dfs)

        if return_ratiometric:
            to_compute.append(movie_r_dask)

        metric_keys = list(metric_tasks.keys())
        for k in metric_keys:
            to_compute.append(metric_tasks[k])

        computed = dask.compute(*to_compute, scheduler=scheduler)

        computed_chunk_dfs = computed[:n_chunks]
        if computed_chunk_dfs:
            df = pd.concat(computed_chunk_dfs, ignore_index=True)
        else:
            df = empty_events_meta.copy()

        comp_idx = n_chunks
        if return_ratiometric:
            movie_r = computed[comp_idx]
            comp_idx += 1
        else:
            movie_r = None

        frame_metrics_dict: dict[str, np.ndarray] = {}
        for k in metric_keys:
            frame_metrics_dict[k] = np.asarray(computed[comp_idx], dtype=np.float32)
            comp_idx += 1

        if compute_movie_metrics:
            if drift_yx_full is not None:
                drift_m = get_drift_metrics(drift_yx_full)
                frame_metrics_dict.update(drift_m)

            n_frames = movie_shape[0]
            frame_metrics_df = pd.DataFrame(frame_metrics_dict, index=np.arange(n_frames))
            frame_metrics_df.index.name = "frame"
        else:
            frame_metrics_df = None

        if frame_metrics_df is not None:
            df.attrs["frame_metrics"] = frame_metrics_df

        df = _apply_postprocess(df, postprocess)

        # Re-attach frame_metrics in case postprocess returned a new DataFrame copy
        if frame_metrics_df is not None and "frame_metrics" not in df.attrs:
            logger.debug("Re-attaching frame_metrics to DataFrame after postprocess.")
            df.attrs["frame_metrics"] = frame_metrics_df

        if return_ratiometric:
            return _apply_analysis_record(
                (df, movie_r),
                attach_record=attach_record,
                func=process,
                local_vars=locals(),
                exclude={"movie", "events"},
                movie_info=movie_info,
            )

        return _apply_analysis_record(
            df,
            attach_record=attach_record,
            func=process,
            local_vars=locals(),
            exclude={"movie", "events"},
            movie_info=movie_info,
        )
