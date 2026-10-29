from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from massphoto.events import annotate_frame_metrics
from massphoto.fitting import fit
from massphoto.movie_drift_correction import get_drift_metrics
from massphoto.movie_process import calc_frame_sharpness
from massphoto.pipeline import process
from massphoto.utils import NativeMovie


def test_calc_frame_sharpness():
    # Synthetic movie with 3 frames
    rng = np.random.default_rng(42)
    # Background + random noise
    base = np.ones((3, 20, 20), dtype=np.float32) * 100.0
    noise = rng.normal(0, 5, size=(3, 20, 20)).astype(np.float32)
    movie = base + noise

    sharpness = calc_frame_sharpness(movie, medfilt_size=5)
    assert sharpness.shape == (3,)
    assert np.all(sharpness > 0)
    assert np.all(np.isfinite(sharpness))


def test_calc_drift_metrics():
    # 5 frames of drift
    drift_yx = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 6.0],
        ],
        dtype=np.float32,
    )

    metrics = get_drift_metrics(drift_yx, fps=10.0)
    assert "drift_y" in metrics
    assert "drift_x" in metrics
    assert "drift_speed" in metrics

    assert np.allclose(metrics["drift_y"], [0.0, 1.0, 1.0, 1.0, 4.0])
    assert np.allclose(metrics["drift_x"], [0.0, 0.0, 2.0, 2.0, 6.0])

    # Step 0: diff is 0 -> speed = 0
    # Step 1: dy=1, dx=0 -> dist=1 -> speed=10
    # Step 2: dy=0, dx=2 -> dist=2 -> speed=20
    # Step 3: dy=0, dx=0 -> dist=0 -> speed=0
    # Step 4: dy=3, dx=4 -> dist=5 -> speed=50
    expected_speeds = [0.0, 10.0, 20.0, 0.0, 50.0]
    assert np.allclose(metrics["drift_speed"], expected_speeds)


def test_annotate_frame_metrics():
    events = pd.DataFrame(
        {
            "frame": [0, 2, 4],
            "y": [10.0, 12.0, 15.0],
            "x": [5.0, 6.0, 7.0],
            "contrast": [0.01, 0.02, 0.015],
        }
    )
    frame_metrics = pd.DataFrame(
        {
            "intensity": [100.0, 101.0, 102.0, 103.0, 104.0],
            "drift_speed": [0.0, 5.0, 10.0, 15.0, 20.0],
        },
        index=[0, 1, 2, 3, 4],
    )

    # Test explicit frame_metrics
    annotated = annotate_frame_metrics(events.copy(), frame_metrics=frame_metrics)
    assert "intensity" in annotated.columns
    assert "drift_speed" in annotated.columns
    assert list(annotated["intensity"]) == [100.0, 102.0, 104.0]
    assert list(annotated["drift_speed"]) == [0.0, 10.0, 20.0]

    # Test via events.attrs
    events_with_attrs = events.copy()
    events_with_attrs.attrs["frame_metrics"] = frame_metrics
    annotated_attrs = annotate_frame_metrics(events_with_attrs)
    assert "intensity" in annotated_attrs.columns
    assert list(annotated_attrs["intensity"]) == [100.0, 102.0, 104.0]

    # Test with column subset
    annotated_subset = annotate_frame_metrics(events.copy(), frame_metrics=frame_metrics, columns=["drift_speed"])
    assert "drift_speed" in annotated_subset.columns
    assert "intensity" not in annotated_subset.columns

    # Test warning when frame_metrics missing
    with pytest.warns(UserWarning, match="No frame_metrics provided or found"):
        events_empty_attrs = events.copy()
        res = annotate_frame_metrics(events_empty_attrs)
        assert "intensity" not in res.columns


def test_pipeline_compute_movie_metrics():
    # 20 frames, 30x30 pixels
    rng = np.random.default_rng(123)
    movie = rng.normal(100.0, 2.0, size=(20, 30, 30)).astype(np.float32)

    # Dummy detection step returning 1 event in first chunk (frame 2)
    def dummy_detection(movie: NativeMovie):
        schema = {
            "frame": pd.Series(dtype=np.int64),
            "y": pd.Series(dtype=np.float64),
            "x": pd.Series(dtype=np.float64),
            "contrast": pd.Series(dtype=np.float64),
        }
        if len(movie) == 0:  # empty movie passed to generate meta
            return pd.DataFrame(schema)
        return pd.DataFrame(
            {
                "frame": [2],
                "y": [15.0],
                "x": [15.0],
                "contrast": [0.01],
            }
        )

    # Process movie with compute_movie_metrics
    result = process(
        movie,
        steps=[(dummy_detection, {})],
        compute_movie_metrics=True,
        movie_correct_drift=True,
        ratiometric_args={"navg": 2, "ngap": 1},
        chunk_overlap=0,
        chunk_size=20,
        postprocess=(annotate_frame_metrics,),
    )

    assert "frame_metrics" in result.attrs
    fm = result.attrs["frame_metrics"]
    assert isinstance(fm, pd.DataFrame)
    assert len(fm) == 20
    assert "intensity" in fm.columns
    assert "sharpness" in fm.columns
    assert "signal" in fm.columns
    assert "drift_y" in fm.columns
    assert "drift_x" in fm.columns
    assert "drift_speed" in fm.columns

    # Check annotated events
    assert "intensity" in result.columns
    assert "drift_speed" in result.columns
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["intensity"], fm.loc[2, "intensity"])


def test_fit_compute_movie_metrics():
    rng = np.random.default_rng(42)
    movie = rng.normal(100.0, 2.0, size=(20, 30, 30)).astype(np.float32)

    # Call mp.fit with compute_movie_metrics
    events = fit(
        movie,
        movie_correct_drift=True,
        ratiometric_args={"navg": 2, "ngap": 1},
        compute_movie_metrics=True,
        postprocess=(annotate_frame_metrics,),
        chunk_size=20,
        progress_bar=False,
    )

    assert "frame_metrics" in events.attrs
    fm = events.attrs["frame_metrics"]
    assert isinstance(fm, pd.DataFrame)
    assert len(fm) == 20
    assert "intensity" in fm.columns
    assert "sharpness" in fm.columns
    assert "signal" in fm.columns
    assert "drift_speed" in fm.columns
