# massphoto Python Package
[![PyPI](https://img.shields.io/pypi/v/massphoto.svg)](https://pypi.org/project/massphoto)
[![Python](https://img.shields.io/pypi/pyversions/massphoto.svg)](https://pypi.org/project/massphoto)
[![License](https://img.shields.io/pypi/l/massphoto.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](https://thielec.github.io/massphoto/)
[![CI](https://img.shields.io/github/actions/workflow/status/thielec/massphoto/ci.yml?branch=main)](https://github.com/thielec/massphoto/actions)

A Python package for processing, analyzing, and quantifying mass photometry data.

* **For mass photometry users:** Analyse your measurements directly in Python, keeping your entire workflow transparent, shareable, and fully under your control.
* **Simple yet flexible:** Turn raw movies into calibrated mass distributions with a single command, or customise parameters like integration time and PSF shape.
* **Open and reproducible:** Automate batch processing across large datasets and output single-molecule events as standard pandas DataFrames to integrate into your analysis pipelines.


## Key Features

- Reading and writing mass photometry data in various formats (MP, TIFF, TDMS)
- Image processing and filtering optimized for mass photometry signals
- Drift correction for measurements
- Event detection using temporal and spatial filters
- PSF fitting for accurate contrast and mass measurements
- Parallel processing with Dask for large datasets

## Getting Started
Install Python 3.10 or higher. Check with `python --version` on Windows or
`python3 --version` on macOS/Linux.

## Installing `massphoto`
Create a new virtual environment and activate it using your shell's commands.

Windows (PowerShell):

```powershell
python -m venv mp-venv
.\mp-venv\Scripts\Activate.ps1
```

For Windows Command Prompt, use `mp-venv\Scripts\activate.bat` instead.

macOS/Linux (bash or zsh):

```bash
python3 -m venv mp-venv
source mp-venv/bin/activate
```

After activation, the following commands work on all platforms. Use `python -m pip`
to install into the active interpreter's environment:

```text
python -m pip install --upgrade pip
python -m pip install massphoto
python -c "import massphoto as mp; print(mp.__version__)"
```

Runtime dependencies, including acceleration and file-format libraries, are installed automatically.

For an editable installation, run this from the source directory containing `pyproject.toml`:

```text
python -m pip install -e .
```

For testing, use `python -m pip install -e ".[testing]"`; for development including
documentation, use `python -m pip install -e ".[dev,docs]"`.

## Upgrading `massphoto`
Activate your environment using the appropriate command above, then run:

```text
python -m pip install --upgrade massphoto
```

The current version can be checked with:
```text
python -m pip show massphoto
```

## Running tests
After installing the testing or development dependencies, run from the repo root:

```text
python -m pytest
```

## Usage Example
In the simplest case, you only call a single function to read a movie file, detect, and fit events:
```python
import massphoto as mp
mpresults = mp.fit("path/to/movie.mp")
```
`mpresults` will be a dataframe with the detected and fitted events.

Common operations stay at the package root. Use `mp.detection` for alternative
detection strategies, `mp.mass_calibration` for calibration, and `mp.distfit` for
distribution fitting:

```python
movie, metadata = mp.read_movie("path/to/movie.mp")
events = mp.fit(movie, detection_func=mp.detection.event_correlation)
distribution = mp.distfit.fit(events["contrasts"])
```

## API Overview

A full API reference is available in the documentation. The main functions include:

### Data I/O
- `read_movie`: Read mass photometry data files (.mp, .mpr, .tiff, .tdms)

### Image Processing
- `calc_ratiometric`: Calculate ratiometric representation
- `normalize_frames`: Normalize native frame means
- `correct_drift`: Correct drift in measurement sequences

### Event Detection
- `detect`: Detect candidate events using chunked execution
- `detection.event_correlation`: Detect using cross-correlation maps (spatial only)
- `detection.event_correlation_step`: Spatial-temporal correlation detection (default detector)

### Event Fitting
- `fit_event`: Fit point spread function to detected events
- `fit`: Full detection and fitting workflow, with configurable chunked execution

### Calibration and Distributions
- `mass_calibration.fit`: Fit a contrast-to-mass calibration
- `mass_calibration.apply`: Apply a calibration to contrasts or an event table
- `distfit.fit`: Fit a mass or contrast distribution and return `distfit.FitResult`

### Event Metrics
- `events.add_nearest_neighbour`: Spatial nearest neighbor distances
- `events.add_local_density`: Local landing density
- `events.annotate_frame_metrics`: Correlate events with frame-level quality metrics

### Custom Pipelines
- `process`: Compose annotated processing steps
- `resolve_record`: Resolve an analysis record into a replay callable
