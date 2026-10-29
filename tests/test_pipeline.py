"""Tests for parallel processing helpers."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from massphoto.detection import event_correlation_step
from massphoto.pipeline import _infer_step_overlap, process, resolve_record
from massphoto.utils import NativeMovie


def _detect_single(movie: NativeMovie) -> pd.DataFrame:
    del movie
    return pd.DataFrame({"frame": [0], "y": [0], "x": [0]})


def test_process_attaches_analysis_record():
    movie = np.zeros((6, 8, 8), dtype=np.float32)
    events = pd.DataFrame({"frame": [0], "y": [0], "x": [0]})

    result = process(
        movie,
        steps=[(_detect_single, {})],
        events=events,
        chunk_size=3,
        chunk_overlap=0,
        progress_bar=False,
    )
    result = cast(pd.DataFrame, result)

    assert "analysis_record" in result.attrs
    record = result.attrs["analysis_record"]
    assert record["schema_version"] == 2
    assert "defaults_state" in record
    assert record["movie_info"]["shape"] == (6, 8, 8)
    assert record["origin"] == "process"
    assert record["origin_kwargs"]["chunk_size"] == 3
    assert "events" not in record["origin_kwargs"]


def test_process_attaches_analysis_record_for_tuple_output():
    movie = np.zeros((4, 8, 8), dtype=np.float32)

    result, movie_r = process(
        movie,
        steps=[(_detect_single, {})],
        return_ratiometric=True,
        ratiometric_func=None,
        chunk_size=2,
        chunk_overlap=0,
        progress_bar=False,
    )

    assert movie_r.shape == movie.shape
    assert "analysis_record" in result.attrs


def test_process_normalizes_frame_means_from_template():
    movie = np.ones((6, 4, 4), dtype=np.float32)
    movie *= np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)[:, None, None]
    template = np.full((4, 4), 10, dtype=np.float32)

    _, normalized = process(
        movie,
        steps=[(_detect_single, {})],
        movie_template=template,
        movie_normalize_frame_means=True,
        return_ratiometric=True,
        ratiometric_func=None,
        chunk_size=2,
        chunk_overlap=0,
        progress_bar=False,
    )

    np.testing.assert_allclose(normalized.mean(axis=(1, 2)), 10)


def test_process_rejects_wrong_template_shape():
    movie = np.ones((6, 4, 4), dtype=np.float32)
    template = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="movie_template must have shape"):
        process(
            movie,
            steps=[(_detect_single, {})],
            movie_template=template,
            movie_normalize_frame_means=True,
            chunk_size=2,
            chunk_overlap=0,
            progress_bar=False,
        )


def test_process_record_kwargs_can_be_reused():
    movie = np.zeros((6, 8, 8), dtype=np.float32)

    first = process(
        movie,
        steps=[(_detect_single, {})],
        chunk_size=3,
        chunk_overlap=0,
        progress_bar=False,
    )
    first = cast(pd.DataFrame, first)
    record = first.attrs["analysis_record"]

    second = process(movie, **record["origin_kwargs"])
    second = cast(pd.DataFrame, second)

    assert first.equals(second)


def test_resolve_record_returns_replay_callable_and_overrides():
    movie = np.zeros((6, 8, 8), dtype=np.float32)
    df = process(
        movie,
        steps=[(_detect_single, {})],
        chunk_size=3,
        chunk_overlap=0,
        progress_bar=False,
    )
    df = cast(pd.DataFrame, df)

    defaults_overrides, replay = resolve_record(df)
    replay_df = replay(movie)
    replay_df = cast(pd.DataFrame, replay_df)

    assert "__meta__" in defaults_overrides
    assert "process.chunk_size" in defaults_overrides
    assert defaults_overrides["process.chunk_size"] == 3
    assert df.equals(replay_df)


def test_resolve_record_accepts_raw_record_dict():
    movie = np.zeros((6, 8, 8), dtype=np.float32)
    df = process(
        movie,
        steps=[(_detect_single, {})],
        chunk_size=3,
        chunk_overlap=0,
        progress_bar=False,
    )
    df = cast(pd.DataFrame, df)
    record = df.attrs["analysis_record"]

    _, replay = resolve_record(record)
    replay_df = replay(movie)
    replay_df = cast(pd.DataFrame, replay_df)
    assert df.equals(replay_df)


def test_resolve_record_roundtrips_via_dataframe_pickle(tmp_path):
    movie = np.zeros((6, 8, 8), dtype=np.float32)
    df = process(
        movie,
        steps=[(_detect_single, {})],
        chunk_size=3,
        chunk_overlap=0,
        progress_bar=False,
    )
    df = cast(pd.DataFrame, df)

    pkl_path = tmp_path / "events.pkl"
    df.to_pickle(pkl_path)
    loaded = pd.read_pickle(pkl_path)

    _, replay = resolve_record(loaded)
    replay_df = cast(pd.DataFrame, replay(movie))
    assert df.equals(replay_df)


def test_process_rejects_invalid_movie_input_type():
    with pytest.raises(TypeError, match="movie must be a dask array"):
        process(cast(Any, 123), steps=[(_detect_single, {})], progress_bar=False)


def test_infer_step_overlap_uses_detection_defaults():
    assert _infer_step_overlap(event_correlation_step, {}) == 20


def test_infer_step_overlap_uses_explicit_detection_kwargs():
    overlap = _infer_step_overlap(
        event_correlation_step,
        {
            "min_distance_temp": 7,
            "threshold_distance_temp": 3,
            "correlation_step_args": {"navg": 11},
        },
    )

    assert overlap == 11
