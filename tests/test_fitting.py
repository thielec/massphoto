"""
Tests for event fitting functionality.
"""

import numpy as np
import pandas as pd
import pytest

import massphoto
from massphoto.detection import detect, event_correlation_step
from massphoto.fitting import fit, fit_event
from massphoto.movie_drift_correction import correct_drift
from massphoto.movie_process import calc_ratiometric
from massphoto.pipeline import resolve_record


def test_fit_low_thresholds_random_movie():
    rng = np.random.default_rng(123)
    movie = (rng.standard_normal((128, 32, 32)) + 100.0).astype(np.float32)
    result = fit(
        movie,
        chunk_size=25,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
        detection_args={
            "thresholds": [0.01, 0.01, 0.0],
            "min_distance_spat": 1,
            "min_distance_temp": 1,
            "output_map_values": True,
        },
    )
    assert isinstance(result, pd.DataFrame)
    assert {"frame", "x", "y"}.issubset(result.columns)


@pytest.mark.parametrize("operation", [fit, detect], ids=["fit", "detect"])
def test_can_disable_analysis_record(operation):
    rng = np.random.default_rng(33)
    movie = (rng.standard_normal((48, 24, 24)) + 100.0).astype(np.float32)

    result = operation(
        movie,
        chunk_size=20,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
        detection_args={
            "thresholds": [0.01, 0.01, 0.0],
            "min_distance_spat": 1,
            "min_distance_temp": 1,
            "output_map_values": True,
        },
        attach_record=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert "analysis_record" not in result.attrs


@pytest.mark.parametrize("operation", [fit, detect], ids=["fit", "detect"])
def test_fit_and_detect_record_origin_and_kwargs(operation):
    rng = np.random.default_rng(45)
    movie = (rng.standard_normal((64, 24, 24)) + 100.0).astype(np.float32)

    result = operation(
        movie,
        chunk_size=20,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
        detection_args={
            "thresholds": [0.01, 0.01, 0.0],
            "min_distance_spat": 1,
            "min_distance_temp": 1,
            "output_map_values": True,
        },
    )

    record = result.attrs["analysis_record"]

    assert record["origin"] == operation.__name__
    assert "events" not in record["origin_kwargs"]


def test_resolve_record_apply_defaults_matches_origin_call():
    rng = np.random.default_rng(46)
    movie = (rng.standard_normal((96, 32, 32)) + 100.0).astype(np.float32)

    fit_result = fit(
        movie,
        chunk_size=24,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
        detection_args={
            "thresholds": [0.01, 0.01, 0.0],
            "min_distance_spat": 1,
            "min_distance_temp": 1,
            "output_map_values": True,
        },
        postprocess=(),
        progress_bar=False,
    )

    with massphoto.defaults.context(copy=True):
        defaults_overrides, replay = resolve_record(fit_result, apply_defaults=True)
        replay_result = replay(movie)
        origin_result = fit(movie)

    replay_result = replay_result.sort_values(["frame", "y", "x"]).reset_index(drop=True)
    origin_result = origin_result.sort_values(["frame", "y", "x"]).reset_index(drop=True)

    assert "fit.chunk_size" in defaults_overrides
    assert defaults_overrides["fit.chunk_size"] == 24
    assert len(replay_result) == len(origin_result)
    assert np.array_equal(
        replay_result[["frame", "x", "y"]].to_numpy(),
        origin_result[["frame", "x", "y"]].to_numpy(),
    )


def test_fit_event_keep_invalid():
    """keep_invalid=True should return one row per candidate, with NaN fit columns for out-of-bounds events."""
    rng = np.random.default_rng(42)
    movie_raw = rng.standard_normal((50, 32, 32)).astype(np.float32) + 100.0
    movie = calc_ratiometric(movie_raw, navg=5)

    # One valid candidate (well inside the frame) and one invalid (at the border,
    # so the thumbnail window would extend outside the movie).
    candidates = pd.DataFrame(
        {
            "frame": [10, 10],
            "y": [15, 0],
            "x": [15, 0],
        }
    )

    result_drop = fit_event(movie, candidates, keep_invalid=False)
    result_keep = fit_event(movie, candidates, keep_invalid=True)

    # keep_invalid=False only returns the valid row
    assert len(result_drop) == 1

    # keep_invalid=True returns all candidates
    assert len(result_keep) == len(candidates)
    assert result_keep.index.tolist() == candidates.index.tolist()

    # Fit columns for the invalid row are NaN
    fit_cols = [
        c for c in result_keep.columns if c not in candidates.columns and c not in ("x_det", "y_det", "contrasts_det")
    ]
    invalid_row = result_keep.loc[candidates.index[1]]
    assert invalid_row[fit_cols].isna().all()

    # Valid row has non-NaN fit values and matches the drop result
    valid_row = result_keep.loc[candidates.index[0]]
    assert not valid_row[fit_cols].isna().any()


def test_fit_with_events_skips_detection():
    rng = np.random.default_rng(1234)
    movie = (rng.standard_normal((96, 32, 32)) + 100.0).astype(np.float32)
    events = pd.DataFrame({"frame": [20], "y": [15], "x": [15]})

    def _failing_detection(*_args, **_kwargs):
        raise AssertionError("detection_func should not be called when events are provided")

    result = fit(
        movie,
        events=events,
        detection_func=_failing_detection,
        chunk_size=25,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert int(result.iloc[0]["frame"]) == 20


def test_fit_with_events_chunked_frames_are_global():
    rng = np.random.default_rng(5678)
    movie = (rng.standard_normal((120, 32, 32)) + 100.0).astype(np.float32)
    events = pd.DataFrame({"frame": [15, 50, 95], "y": [15, 16, 14], "x": [15, 16, 14]})

    result = fit(
        movie,
        events=events,
        chunk_size=30,
        chunk_overlap=3,
        ratiometric_args={"navg": 2, "ngap": 1},
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result["frame"].astype(int).tolist()) == {15, 50, 95}


def test_fit_with_events_missing_columns_raises():
    rng = np.random.default_rng(9)
    movie = rng.standard_normal((40, 32, 32)).astype(np.float32)
    invalid_events = pd.DataFrame({"frame": [10], "y": [15]})

    with pytest.raises(ValueError, match="frame"):
        fit(movie, events=invalid_events, ratiometric_args={"navg": 2, "ngap": 1})


def test_fit_with_events_invalid_array_shape_raises():
    rng = np.random.default_rng(10)
    movie = rng.standard_normal((40, 32, 32)).astype(np.float32)
    invalid_events = [[10, 15, 15, 1.0], [20, 16, 16, 2.0]]

    with pytest.raises(ValueError, match="shape"):
        fit(movie, events=invalid_events, ratiometric_args={"navg": 2, "ngap": 1})


def test_fit_with_events_nx3_array_is_accepted():
    rng = np.random.default_rng(11)
    movie = (rng.standard_normal((96, 32, 32)) + 100.0).astype(np.float32)
    events_nx3 = np.array(
        [
            [20, 15, 15],
            [40, 16, 16],
        ]
    )

    result = fit(
        movie,
        events=events_nx3,
        chunk_size=25,
        chunk_overlap=0,
        ratiometric_args={"navg": 2, "ngap": 1},
    )
    if isinstance(result, tuple):
        result = result[0]

    assert isinstance(result, pd.DataFrame)
    assert set(result["frame"].astype(int).tolist()) == {20, 40}


def test_fit_with_events_3xn_array_raises():
    rng = np.random.default_rng(12)
    movie = rng.standard_normal((96, 32, 32)).astype(np.float32)
    events_3xn = np.array(
        [
            [20, 40],
            [15, 16],
            [15, 16],
        ]
    )

    with pytest.raises(ValueError, match="shape"):
        fit(movie, events=events_3xn, ratiometric_args={"navg": 2, "ngap": 1})


def test_fit_with_events_1d_array_raises():
    rng = np.random.default_rng(13)
    movie = rng.standard_normal((96, 32, 32)).astype(np.float32)
    events_1d = np.array([20, 15, 15])

    with pytest.raises(ValueError, match="2D"):
        fit(movie, events=events_1d, ratiometric_args={"navg": 2, "ngap": 1})


def test_fit_matches_two_step_pipeline(frames):
    """Running detect+fit in one call should match detect then fit(events=...)."""
    movie = frames[:300]
    ratiometric_args = {"navg": 10, "ngap": 1}
    chunk_size = 100

    fit_one_step = fit(
        movie,
        movie_correct_drift=True,
        ratiometric_args=ratiometric_args,
        chunk_size=chunk_size,
        postprocess=(),
        scheduler="threads",
        progress_bar=False,
    )

    events = detect(
        movie,
        movie_correct_drift=True,
        ratiometric_args=ratiometric_args,
        chunk_size=chunk_size,
        scheduler="threads",
        progress_bar=False,
    )

    fit_two_step = fit(
        movie,
        events=events,
        movie_correct_drift=True,
        ratiometric_args=ratiometric_args,
        chunk_size=chunk_size,
        postprocess=(),
        scheduler="threads",
        progress_bar=False,
    )

    fit_one_step = fit_one_step.sort_values(["frame", "y", "x"]).reset_index(drop=True)
    fit_two_step = fit_two_step.sort_values(["frame", "y", "x"]).reset_index(drop=True)

    assert len(fit_one_step) > 0
    assert len(fit_one_step) == len(fit_two_step)
    assert np.array_equal(fit_one_step[["frame", "x", "y"]].to_numpy(), fit_two_step[["frame", "x", "y"]].to_numpy())
    assert np.allclose(fit_one_step["contrasts"].to_numpy(), fit_two_step["contrasts"].to_numpy(), atol=1e-12, rtol=0.0)


def test_fit_matches_manual_pipeline(frames):
    """Parallel one-step pipeline should match manual drift->ratiometric->detect->fit processing."""
    movie = frames[:300]
    ratiometric_args = {"navg": 10, "ngap": 1}

    fit_one_step = fit(
        movie,
        movie_correct_drift=True,
        ratiometric_args=ratiometric_args,
        chunk_size=1000,
        postprocess=(),
        scheduler="threads",
        progress_bar=False,
    )

    movie_corrected = correct_drift(movie, template=np.mean(movie, axis=0))
    movie_ratiometric = calc_ratiometric(movie_corrected, **ratiometric_args)
    events_manual = event_correlation_step(movie_corrected, movie_ratiometric)
    fit_manual = fit_event(movie_ratiometric, events_manual)

    fit_one_step = fit_one_step.sort_values(["frame", "y", "x"]).reset_index(drop=True)
    fit_manual = fit_manual.sort_values(["frame", "y", "x"]).reset_index(drop=True)

    assert len(fit_one_step) > 0
    assert len(fit_one_step) == len(fit_manual)
    assert np.array_equal(
        fit_one_step[["frame", "x", "y"]].to_numpy(),
        fit_manual[["frame", "x", "y"]].to_numpy(),
    )
    assert np.allclose(
        fit_one_step["contrasts"].to_numpy(),
        fit_manual["contrasts"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )
