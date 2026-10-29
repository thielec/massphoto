"""Public beta imports, configuration, and replay contracts."""

import importlib

import numpy as np
import pandas as pd
import pytest

import massphoto as mp


def _detect_single(movie: mp.NativeMovie) -> pd.DataFrame:
    return pd.DataFrame({"frame": [0], "y": [0], "x": [0]})


def _fit_events(events: mp.EventList) -> pd.DataFrame:
    return events.assign(contrasts=0.001)


def test_root_exports():
    expected = {
        "__version__",
        "defaults",
        "read_movie",
        "save_mp",
        "save_h5",
        "save_tiff",
        "calc_ratiometric",
        "calc_differential",
        "calc_mobileframes",
        "normalize_frames",
        "rebin_movie",
        "correct_drift",
        "detect",
        "fit",
        "detect_on_map",
        "fit_event",
        "get_psf_thumbnails",
        "process",
        "resolve_record",
        "NativeMovie",
        "RatiometricMovie",
        "EventList",
        "convert_metadata",
        "add_local_density",
        "add_nearest_neighbour",
        "annotate_frame_metrics",
        "calc_frame_sharpness",
        "PSF2D",
        "fit_psf_params",
        "filter",
        "detection",
        "distfit",
        "events",
        "mass_calibration",
    }
    assert set(mp.__all__) == expected
    assert len(mp.__all__) == len(expected)
    assert all(hasattr(mp, name) for name in expected)


def test_namespace_functions_survive_implementation_imports():
    fitting = importlib.import_module("massphoto.fitting")
    pipeline = importlib.import_module("massphoto.pipeline")
    distribution = importlib.import_module("massphoto.distfit.fitting")
    utilities = importlib.import_module("massphoto.utils")
    assert mp.fit is fitting.fit
    assert mp.detect is mp.detection.detect
    assert mp.process is pipeline.process
    assert mp.distfit.fit is distribution.fit
    assert mp.distfit.FitResult is distribution.FitResult
    assert mp.defaults is utilities.defaults
    assert callable(mp.mass_calibration.apply)
    assert callable(mp.mass_calibration.fit)
    assert callable(mp.detection.event_correlation_step)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 1}, "Unsupported.*schema_version"),
        ({"schema_version": 999}, "Unsupported.*schema_version"),
        ({"origin": "fit_parallel"}, "Unknown record origin"),
        ({"origin": "detect_parallel"}, "Unknown record origin"),
        ({"origin": "process_parallel"}, "Unknown record origin"),
        ({"origin": "unknown"}, "Unknown record origin"),
    ],
)
def test_invalid_records_do_not_apply_defaults(change, message):
    record = {
        "schema_version": 2,
        "origin": "fit",
        "defaults_state": {"invalid_record_sentinel": True},
        "origin_kwargs": {},
    }
    record.update(change)
    with mp.defaults.context(copy=True):
        with pytest.raises(ValueError, match=message):
            mp.resolve_record(record, apply_defaults=True)
        assert mp.defaults.get("invalid_record_sentinel") is None


@pytest.mark.parametrize("missing", ["schema_version", "origin", "defaults_state", "origin_kwargs"])
def test_record_requires_schema_and_origin(missing):
    record = {"schema_version": 2, "origin": "process", "defaults_state": {}, "origin_kwargs": {}}
    del record[missing]
    with pytest.raises(ValueError, match=f"missing required keys: {missing}"):
        mp.resolve_record(record)


def test_pipeline_with_process_scheduler():
    result = mp.process(
        np.ones((4, 4, 4)),
        steps=[(_detect_single, {})],
        chunk_size=2,
        chunk_overlap=0,
        scheduler="processes",
        progress_bar=False,
    )
    assert result["frame"].tolist() == [0, 2]
