"""
Metadata utilities.

The public helpers in this module normalise instrument metadata into a single
dictionary with a few well-known keys such as ``framerate``, ``pixelsize`` and
``exposuretime``.

Converted metadata schema (typical)::

    {
        "framerate": value,           # float, value before any binning
        "framerate_unit": value,      # str, typically "Hz"
        "framebinning": value,        # int, number of frames binned together
        "pixelsize": value,           # float, value before any binning
        "pixelsize_unit": value,      # str, typically "um"
        "pixelbinning": value,        # int, number of pixels binned together
        "exposuretime": value,        # float, value before any binning
        "exposuretime_unit": value,   # str, typically "ms"
        "instrument": value,          # str, instrument name (if available)
        "camera": value,              # str, camera name (if available)
    }
"""

import logging

logger = logging.getLogger(__name__)


def empty_metadata() -> dict:
    """
    Return minimal, generic metadata.

    Returns
    -------
    dict
        Metadata dictionary with default values for framerate, pixelsize and
        exposuretime.
    """
    return {
        "framerate": 1,
        "framerate_unit": "1/frame",
        "framebinning": 1,
        "pixelsize": 1.0,
        "pixelsize_unit": "px",
        "pixelbinning": 1,
        "exposuretime": 1.0,
        "exposuretime_unit": "frame",
        "instrument": "unknown",
        "camera": "unknown",
    }


def update_derived_metadata(metadata: dict) -> dict:
    """
    Update derived fields for standardized metadata.

    Parameters
    ----------
    metadata
        Standardized metadata dictionary.

    Returns
    -------
    dict
        The same dictionary with derived fields refreshed.
    """
    required_keys = ("framerate", "framebinning", "pixelsize", "pixelbinning", "image_shape")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise ValueError(
            "metadata must be standardized (use massphoto.metadata.convert_metadata) and include keys: "
            f"{required_keys}. Missing keys: {missing_keys}."
        )

    metadata["framerate_effective"] = metadata["framerate"] / metadata["framebinning"]
    metadata["pixelsize_effective"] = metadata["pixelsize"] * metadata["pixelbinning"]

    if "exposuretime" in metadata:
        metadata["exposuretime_effective"] = metadata["exposuretime"] * metadata["framebinning"]

    metadata["field_of_view"] = (
        metadata["image_shape"][0] * metadata["pixelsize_effective"],
        metadata["image_shape"][1] * metadata["pixelsize_effective"],
    )
    if "pixelsize_unit" in metadata:
        metadata["field_of_view_unit"] = metadata["pixelsize_unit"]

    return metadata


def _numpy_to_python(value):
    if isinstance(value, tuple):
        return tuple(_numpy_to_python(v) for v in value)
    return value.item() if hasattr(value, "item") else value


INSTRUMENT_PIXEL_SIZES: dict[str, float] = {
    "Refeyn OneMP": 0.0193,
    "Refeyn TwoMP": 0.0118,
}


def _decode_str(val: object, default: str = "unknown") -> str:
    """Safely decode metadata bytes or string."""
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _extract_raw_fields(metadata: dict) -> dict:
    """Extract format-specific raw acquisition fields from metadata."""
    if metadata.get("format_version_number", 0) >= 3:
        # Refeyn MP v3+ (Variant A: AcqCam)
        if metadata.get("movie/configuration/Devices/AcqCam/Height") is not None:
            binning = metadata["movie/configuration/Devices/AcqCam/SoftwarePixelBinning"]
            h = metadata["movie/configuration/Devices/AcqCam/Height"] // binning
            w = metadata["movie/configuration/Devices/AcqCam/Width"] // binning
            return {
                "framerate": metadata["movie/configuration/Devices/AcqCam/FrameRate"],
                "framebinning": metadata["movie/configuration/Devices/AcqCam/SoftwareFrameBinning"],
                "pixelbinning": binning,
                "image_shape": (h, w),
                "exposuretime": metadata["movie/configuration/Devices/AcqCam/ExposureTime"],
                "instrument": _decode_str(metadata.get("movie/device_info/InstrumentName")),
                "camera": _decode_str(metadata.get("movie/configuration/Devices/AcqCam/CameraName")),
            }
        # Refeyn MP v3+ (Variant B: acq_camera)
        return {
            "framerate": metadata["movie/configuration/acq_camera/frame_rate"],
            "framebinning": metadata["movie/configuration/acq_camera/frame_binning"],
            "pixelbinning": metadata["movie/configuration/acq_camera/pixel_binning"],
            "image_shape": tuple(metadata["movie/keyframe"].shape[1:]),
            "exposuretime": metadata["movie/configuration/acq_camera/exposure_time"],
            "instrument": _decode_str(metadata.get("movie/device_serials/InstrumentName")),
            "camera": _decode_str(metadata.get("movie/configuration/acq_camera/model")),
        }

    # TDMS / custom format
    if metadata.get("Image size") is not None:
        return {
            "framerate": metadata["Frame rate ( before binning )"],
            "framebinning": metadata["Bin frame size"],
            "pixelbinning": metadata["Pixel Binning"],
            "image_shape": (metadata["Image size"], metadata["Image size 2"]),
            "exposuretime": metadata["Exposure Time"] / 1000.0,
            "instrument": "custom",
            "camera": "unknown",
        }

    # Refeyn MP v2
    binning = metadata["configuration/Engines/AcqMovieEngine/PixelBinning"]
    h = metadata["configuration/Devices/AcqCam/Height"] // binning
    w = metadata["configuration/Devices/AcqCam/Width"] // binning
    return {
        "framerate": metadata["configuration/Devices/AcqCam/FrameRate"],
        "framebinning": metadata["configuration/Engines/AcqMovieEngine/FrameBinning"],
        "pixelbinning": binning,
        "image_shape": (h, w),
        "exposuretime": metadata["configuration/Devices/AcqCam/ExposureTime"],
        "instrument": _decode_str(metadata.get("device_info/InstrumentName")),
        "camera": _decode_str(metadata.get("configuration/Devices/AcqCam/CameraName")),
    }


def convert_metadata(metadata: dict, include_native: bool = False) -> dict:
    """
    Convert metadata dictionary to a standardized format.

    Parameters
    ----------
    metadata
        Metadata dictionary to convert.
    include_native
        Include the original input mapping under a separate key in the output.

    Returns
    -------
    dict
        Normalised metadata dictionary.
    """
    if not metadata:
        return empty_metadata()

    try:
        raw = _extract_raw_fields(metadata)
        out_dict = empty_metadata()
        out_dict.update(
            {
                "framerate_unit": "Hz",
                "exposuretime_unit": "ms",
                **raw,
            }
        )
        lookup_pixelsize(out_dict)
        update_derived_metadata(out_dict)
        out_dict = {k: _numpy_to_python(v) for k, v in out_dict.items()}

        if include_native:
            out_dict["native"] = metadata
    except Exception as e:
        logger.warning("Error converting metadata: %s", e)
        out_dict = empty_metadata()
    return out_dict


def lookup_pixelsize(metadata: dict) -> float | None:
    """
    Look up and apply known instrument pixel sizes in micrometers.
    """
    instrument = metadata.get("instrument")
    if instrument in INSTRUMENT_PIXEL_SIZES:
        pixelsize = INSTRUMENT_PIXEL_SIZES[instrument]
        metadata["pixelsize"] = pixelsize
        metadata["pixelsize_unit"] = "um"
        return pixelsize
    return None
