"""
This module contains functions to calculate additional information and metrics about event localisations including:
 - the distance between events and their nearest neighbour
 - the local density of events
 - the local average contrast of events
 - the intersection/Jaccard index of two sets of points
 - detection efficiency and accuracy of predicted points if ground truth points are provided
"""

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy as sp

logger = logging.getLogger(__name__)


def distance_nearest_neighbour(
    a: np.ndarray,
    b: np.ndarray | None = None,
    r: float = float("inf"),
    return_idx: bool = False,
) -> np.ndarray:
    """
    Calculate the distance between each point in a and its nearest neighbour in b.

    Parameters:
    a (array-like, Kxn): First set of nD points.
    b (array-like, Kxn): Second set of nD points. If None, b is set to a. Default is None.
    r (float): Cutoff distance for considering points as neighbours. Default is inf.
    return_idx (bool): Whether to return the index of the nearest neighbour. Default is False.

    Returns:
    array-like: An array of the same length as a, containing the distance to the nearest neighbour in b for each point in a.
    """
    if b is None:
        # Build KDTree for set a
        tree_a = sp.spatial.KDTree(a)
        # Query the tree for each point in a (excluding itself)
        distance, idx = tree_a.query(a, k=(2,), distance_upper_bound=r)
    else:
        # Build KDTree for set b
        tree_b = sp.spatial.KDTree(b)
        # Query the tree for each point in a
        distance, idx = tree_b.query(a, k=1, distance_upper_bound=r)
    if return_idx:
        return distance, idx
    return distance


def points_intersect(a: np.ndarray, b: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    Calculate the intersection of two sets of nD points.

    Parameters:
    a (array-like, Kxn): First set of nD points.
    b (array-like, Kxn): Second set of nD points.
    r (float): Tolerance parameter for considering points as similar. Default is 1.0.

    Returns:
    array-like: A boolean array of the same length as a, indicating whether the corresponding point in a is within r of any point in b.
    """
    # Returns inf if no point is within r
    within_tolerance = distance_nearest_neighbour(a, b, r)
    # If the distance is finite, the point is within r
    return np.isfinite(within_tolerance)


def jaccard_index_tree(a: np.ndarray, b: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    Calculate the Jaccard index between two sets of 2D points.

    Parameters:
    a (array-like, Nx2): First set of 2D points.
    b (array-like, Nx2): Second set of 2D points.
    r (float): Tolerance parameter for considering points as similar. Default is 1.0.

    Returns:
    float: Jaccard index, a value between 0 and 1 representing the similarity between the two sets.
    """
    within_tolerance = points_intersect(a, b, r)
    # Calculate the Jaccard index
    jaccard = np.mean(within_tolerance)
    return jaccard


def detection_stats(gt: np.ndarray, pred: np.ndarray, r: float = 1.0) -> tuple[float, float]:
    """Calculate detection recall and precision given ground-truth and predicted positions.

    Evaluates the fraction of ground truth points recovered (recall / detection efficiency)
    and the fraction of predicted points that match ground truth (precision / accuracy)
    within a distance threshold ``r``.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth point coordinates of shape (N, 2).
    pred : np.ndarray
        Predicted point coordinates of shape (M, 2).
    r : float, default=1.0
        Tolerance radius for considering points matched.

    Returns
    -------
    tuple[float, float]
        (recall, precision) representing the detection efficiency and accuracy.
    """
    return jaccard_index_tree(gt, pred, r), jaccard_index_tree(pred, gt, r)


def filter_nearest_neighbour(
    locs: pd.DataFrame,
    r_min: float = 5.0,
    ratio_temporal_spatial: float = 1.0,
    return_df: bool = True,
) -> np.ndarray | pd.DataFrame:
    """
    Filter out points that are too close to their nearest neighbour.

    This function takes a DataFrame of localizations and filters out points that have a nearest
    neighbor within a minimum distance (r_min). The distance calculation considers both spatial
    and temporal coordinates, where the temporal coordinate can be weighted differently using
    the ratio_temporal_spatial parameter.

    Parameters
    ----------
    locs : pd.DataFrame
        DataFrame containing localizations with at least 'frame', 'y', and 'x' columns.
    r_min : float, optional
        Minimum allowed distance to nearest neighbour. Default is 5.0.
    ratio_temporal_spatial : float, optional
        Scaling factor for temporal dimension relative to spatial dimensions. Default is 1.0.
    return_df : bool, optional
        If True, returns filtered DataFrame. If False, returns boolean mask. Default is True.

    Returns
    -------
    np.ndarray or pd.DataFrame
        If return_df is True, returns filtered DataFrame containing only points that are
        farther than r_min from their nearest neighbour.
        If return_df is False, returns boolean mask where True indicates points that pass
        the distance filter.
    """
    locs_tyx = locs[["frame", "y", "x"]].to_numpy().astype(float)
    locs_tyx[:, 0] *= ratio_temporal_spatial
    # Calculate the distance to the nearest neighbour
    distance = distance_nearest_neighbour(locs_tyx, r=r_min)
    # Filter out points that are too close to their nearest neighbour
    ind = distance > r_min
    if return_df:
        return locs[ind]
    return ind


def add_nearest_neighbour(
    locs: pd.DataFrame,
    r_max: float = 10.0,
    ratio_temporal_spatial: float = 1.0,
) -> pd.DataFrame:
    """
    Add the nearest neighbour distance to the localizations DataFrame.

    This function takes a DataFrame of localizations and adds a new column
    containing the distance to the nearest neighbour for each point. The distance
    calculation considers both spatial and temporal coordinates, where the temporal
    coordinate can be weighted differently using the ratio_temporal_spatial parameter.

    Parameters
    ----------
    locs : pd.DataFrame
        DataFrame containing localizations with at least 'frame', 'y', and 'x' columns.
    r_max : float, optional
        Search radius for nearest neighbour. Default is 10.0. Larger values may increase computation time.
    ratio_temporal_spatial : float, optional
        Scaling factor for temporal dimension relative to spatial dimensions. Default is 1.0. Increase for long average times.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new column 'nn_dist' containing the distance to the nearest neighbour.
    """
    locs_tyx = locs[["frame", "y", "x"]].to_numpy().astype(float)
    locs_tyx[:, 0] *= ratio_temporal_spatial
    # Calculate the distance to the nearest neighbour
    distance = distance_nearest_neighbour(locs_tyx, r=r_max)
    # Add the distance as a new column in the DataFrame
    locs["nn_dist"] = distance
    return locs


def add_local_density(
    locs: pd.DataFrame,
    r_smooth: float = 1.0,
    resolution: float = 1.0,
    filter_str: str | None = None,
    return_map: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Add the local density to the localizations DataFrame.

    This function computes a local density estimate map and looks up the density
    for each localization. The local density is estimated by binning spatial
    coordinates into a 2D histogram and smoothing with a Gaussian filter.
    The result is added as a new column 'local_density' in the DataFrame.
    Optionally, the smoothed density map can also be returned.

    Parameters
    ----------
    locs : pd.DataFrame
        DataFrame containing localizations with at least 'frame', 'y', and 'x' columns.
    r_smooth : float, optional
        Standard deviation (in pixels) for Gaussian smoothing of the density map. Default is 1.0.
    resolution : float, optional
        Bin size for the histogram in pixels. Default is 1.0.
    filter_str : str or None, optional
        Optional pandas query string to filter localizations before density calculation.
        Default is None (all finite localizations are included).
    return_map : bool, optional
        If True, also returns the smoothed density map as a numpy array. Default is False.

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, np.ndarray)
        DataFrame with a new column 'local_density' containing the local density for each localization.
        If return_map is True, also returns the smoothed density map as a numpy array.

    Notes
    -----
    - No correction for edge effects is applied; densities near the edge may be underestimated. Especially
        for large r_smooth values.
    - The density is normalized by the number of frames. The units are events/pixel^2/frame.
    """
    locs["local_density"] = np.nan
    locs_tyx = locs[["frame", "y", "x"]].to_numpy(dtype=float)
    ind_valid = np.isfinite(locs_tyx).all(axis=1)
    if filter_str is not None and filter_str != "":
        ind_valid &= np.asarray(locs.eval(filter_str), dtype=bool)
    if not np.any(ind_valid):
        return locs
    locs_tyx = locs_tyx[ind_valid]
    min_tyx = np.min(locs_tyx, axis=0)
    max_tyx = np.max(locs_tyx, axis=0)
    range_tyx = max_tyx - min_tyx + 1.0
    # subtract min to get zero-based coordinates
    locs_tyx -= min_tyx
    # build a 2d histogram of localizations with pixel/ resolution precision
    x_bins = np.arange(0, max_tyx[2] - min_tyx[2] + 2 * resolution, resolution, dtype=float)
    y_bins = np.arange(0, max_tyx[1] - min_tyx[1] + 2 * resolution, resolution, dtype=float)

    logger.debug("locs_tyx shape: %s", locs_tyx.shape)
    logger.debug(
        "locs_tyx x-range: min=%s max=%s bins=(%s, %s)",
        np.min(locs_tyx[:, 2]),
        np.max(locs_tyx[:, 2]),
        x_bins[0],
        x_bins[-1],
    )
    logger.debug(
        "locs_tyx y-range: min=%s max=%s bins=(%s, %s)",
        np.min(locs_tyx[:, 1]),
        np.max(locs_tyx[:, 1]),
        y_bins[0],
        y_bins[-1],
    )

    loc_map, _, _ = np.histogram2d(
        locs_tyx[:, 1],
        locs_tyx[:, 2],
        bins=[y_bins, x_bins],
    )
    # smooth the localization map with a gaussian filter
    sigma_px = r_smooth / resolution
    loc_map_smooth = sp.ndimage.gaussian_filter(loc_map, sigma=sigma_px)
    # normalize to get density per pixel per frame
    loc_map_smooth /= resolution**2 * range_tyx[0]
    # look up the density for each localization
    locs_density = loc_map_smooth[
        np.round(locs_tyx[:, 1] / resolution).astype(int), np.round(locs_tyx[:, 2] / resolution).astype(int)
    ]
    locs.loc[ind_valid, "local_density"] = locs_density
    if return_map:
        return locs, loc_map_smooth
    return locs


def remove_duplicates(df, merge_radius=2, precedence="-index", iterative=True, ratio_temporal_spatial=1.0):
    """
    Removes duplicate detections in the DataFrame based on the 3D (frame, y, x) distance.

    This function identifies detections that are within a specified radius of each other
    (in both space and time), and merges them by keeping only one detection per group of duplicates.
    The detection to keep is determined by the `precedence` argument, which can be any valid
    pandas DataFrame expression (e.g., '-index', 'contrasts', etc.).
    The temporal dimension can be weighted relative to the spatial dimensions using the
    `ratio_temporal_spatial` parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing detections with at least 'frame', 'y', and 'x' columns.
    merge_radius : float, optional
        Maximum allowed distance (in scaled 3D space) for detections to be considered duplicates. Default is 2.
    precedence : str, optional
        Expression to determine which detection to keep among duplicates. Default is '-index' (keep lowest index).
    iterative : bool, optional
        Whether to iteratively merge duplicates until no more are found. Default is True.
    ratio_temporal_spatial : float, optional
        Scaling factor for the temporal ('frame') dimension relative to spatial ('y', 'x') dimensions. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate detections merged according to the specified criteria.

    Notes
    -----
    - The function uses a KDTree to efficiently find nearby detections in 3D (frame, y, x) space.
    - If `iterative` is True, the merging process is repeated up to 10 times or until no further merges occur.
    - The `precedence` argument is evaluated using pandas `.eval()` and can reference any column in the DataFrame.
    """
    if df.shape[0] == 0:
        return df.copy()
    if iterative:
        MAX_ITER = 10
        for _ in range(MAX_ITER):
            last_shape = df.shape[0]
            df = remove_duplicates(
                df,
                merge_radius=merge_radius,
                precedence=precedence,
                iterative=False,
                ratio_temporal_spatial=ratio_temporal_spatial,
            )
            if df.shape[0] == last_shape:
                break
        return df

    coords = df[["frame", "y", "x"]].to_numpy().astype(float)
    coords[:, 0] *= ratio_temporal_spatial
    _, idx = distance_nearest_neighbour(coords, None, r=merge_radius, return_idx=True)
    idx = idx[:, 0]
    ind_unq = df.shape[0] <= idx
    # calculate the metric for the precedence
    precedence_ind = df[~ind_unq].eval(precedence).to_numpy() > df.iloc[idx[~ind_unq]].eval(precedence).to_numpy()
    ind_unq[~ind_unq] = precedence_ind
    return df[ind_unq].copy()


def annotate_frame_metrics(
    events: pd.DataFrame,
    frame_metrics: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Annotate an event DataFrame with frame-level metrics.

    Parameters
    ----------
    events : pd.DataFrame
        Event DataFrame containing a ``frame`` column.
    frame_metrics : pd.DataFrame | None, optional
        Frame-indexed DataFrame containing metrics. If ``None``, it is retrieved
        from ``events.attrs['frame_metrics']``.
    columns : Sequence[str] | None, optional
        Subset of columns from ``frame_metrics`` to add to ``events``. If ``None``,
        all available metric columns are mapped.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with metric columns added.
    """
    if frame_metrics is None:
        frame_metrics = events.attrs.get("frame_metrics")

    if frame_metrics is None:
        warnings.warn("No frame_metrics provided or found in events.attrs; cannot annotate events.", stacklevel=2)
        return events

    if events.empty or "frame" not in events.columns:
        return events

    cols_to_map = list(columns) if columns is not None else list(frame_metrics.columns)
    for col in cols_to_map:
        if col in frame_metrics.columns:
            events[col] = events["frame"].map(frame_metrics[col])

    return events
