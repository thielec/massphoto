"""
Tests for events module functions.
"""

import numpy as np
import pandas as pd

from massphoto.events import (
    add_local_density,
    add_nearest_neighbour,
    detection_stats,
    distance_nearest_neighbour,
    filter_nearest_neighbour,
    jaccard_index_tree,
    points_intersect,
    remove_duplicates,
)


def test_distance_nearest_neighbour_same_points():
    """Test distance calculation when a is used for both sets."""
    points = np.array([[0, 0], [3, 0], [0, 4]])

    # When b is None, it should return distance to the second closest point
    distances = distance_nearest_neighbour(points).flatten()
    expected = np.array([3, 3, 4])
    np.testing.assert_array_almost_equal(distances, expected)

    # Test index return
    distances, indices = distance_nearest_neighbour(points, return_idx=True)
    expected_indices = np.array([1, 0, 0])
    np.testing.assert_array_equal(indices.flatten(), expected_indices.flatten())


def test_distance_nearest_neighbour_different_points():
    """Test distance calculation between two different sets of points."""
    points_a = np.array([[1, 1], [5, 5]])
    points_b = np.array([[2, 2], [10, 10]])

    distances = distance_nearest_neighbour(points_a, points_b)

    # Expected: distance from [1,1] to [2,2] is sqrt(2), and from [5,5] to [2,2] is sqrt(18)
    expected = np.array([np.sqrt(2), np.sqrt(18)])
    np.testing.assert_array_almost_equal(distances, expected)


def test_distance_nearest_neighbour_with_cutoff():
    """Test distance calculation with a cutoff radius."""
    points_a = np.array([[0, 0], [10, 10]])
    points_b = np.array([[1, 1], [20, 20]])

    # Set a cutoff distance of 2, meaning only neighbors within distance 2 will be returned
    # For distances > 2, it should return inf
    distances = distance_nearest_neighbour(points_a, points_b, r=2)

    # [0,0] is within r=2 of [1,1], but [10,10] has no neighbors within r=2
    expected = np.array([np.sqrt(2), np.inf])
    np.testing.assert_array_almost_equal(distances, expected)


def test_points_intersect():
    """Test the identification of intersecting points."""
    points_a = np.array([[0, 0], [5, 5], [10, 10]])
    points_b = np.array([[1, 1], [6, 6], [20, 20]])

    # With r=2, the first two points in a are within r of points in b
    intersections = points_intersect(points_a, points_b, r=2)

    expected = np.array([True, True, False])
    np.testing.assert_array_equal(intersections, expected)


def test_jaccard_index_tree():
    """Test the calculation of Jaccard index between two point sets."""
    points_a = np.array([[0, 0], [5, 5], [10, 10]])
    points_b = np.array([[1, 1], [6, 6], [20, 20]])

    # With r=2, 2 of 3 points in a are close to points in b
    jaccard = jaccard_index_tree(points_a, points_b, r=2)

    # Expected: 2/3
    assert np.isclose(jaccard, 2 / 3)


def test_detection_stats():
    """Test the calculation of detection efficiency and accuracy."""
    ground_truth = np.array([[0, 0], [5, 5], [10, 10], [15, 15]])
    predicted = np.array([[1, 1], [6, 6], [20, 20]])

    efficiency, accuracy = detection_stats(ground_truth, predicted, r=2)

    # Efficiency: 2/4 points in ground_truth detected
    # Accuracy: 2/3 points in predicted are ground_truth
    assert np.isclose(efficiency, 2 / 4)
    assert np.isclose(accuracy, 2 / 3)


def test_filter_nearest_neighbour():
    """Test filtering of points based on nearest neighbor distance."""
    # Create a DataFrame with points that are both close and far from each other
    df = pd.DataFrame({"frame": [1, 1, 2, 2], "y": [10, 12, 20, 30], "x": [10, 11, 20, 30]})

    # Filter with minimum distance 5 - should keep points that are far enough apart
    # The first two points are too close to each other (distance ~sqrt(5))
    filtered_df = filter_nearest_neighbour(df, r_min=5.0)

    # Only the points at (20,20) and (30,30) should remain as they're far enough apart
    assert len(filtered_df) == 2
    assert filtered_df["frame"].iloc[0] == 2.0
    assert filtered_df["y"].iloc[0] == 20.0

    # Test with return_df=False to get the mask
    mask = filter_nearest_neighbour(df, r_min=5.0, return_df=False)
    expected_mask = np.array([False, False, True, True])
    np.testing.assert_array_equal(mask.flatten(), expected_mask.flatten())


def test_add_nearest_neighbour_distance_column():
    points = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "y": [0, 0, 4],
            "x": [0, 3, 8],
        }
    )
    result = add_nearest_neighbour(points.copy(), r_max=10.0)
    assert "nn_dist" in result.columns
    coords = points[["frame", "y", "x"]].to_numpy(dtype=float, copy=True)
    coords[:, 0] *= 1.0
    expected = distance_nearest_neighbour(coords, r=10.0)
    np.testing.assert_array_almost_equal(result["nn_dist"].to_numpy(), expected.flatten())


def test_add_local_density_with_map():
    positions = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "y": [0, 0, 1],
            "x": [0, 1, 1],
        }
    )
    result, density_map = add_local_density(positions.copy(), r_smooth=0.5, resolution=1.0, return_map=True)
    assert "local_density" in result.columns
    locs_tyx = positions[["frame", "y", "x"]].to_numpy(dtype=float, copy=True)
    min_tyx = np.min(locs_tyx, axis=0)
    locs_tyx -= min_tyx
    indices_y = np.round(locs_tyx[:, 1]).astype(int)
    indices_x = np.round(locs_tyx[:, 2]).astype(int)
    expected_density = density_map[indices_y, indices_x]
    np.testing.assert_allclose(result["local_density"].to_numpy(), expected_density)
    assert np.all(np.isfinite(result["local_density"]))
    mid_density = result["local_density"].iloc[1]
    other_densities = result["local_density"].drop(index=1)
    assert np.all(mid_density > other_densities)

    wide_smooth = add_local_density(positions.copy(), r_smooth=2.0, resolution=1.0)
    assert np.nanmax(result["local_density"]) > np.nanmax(wide_smooth["local_density"])


def test_remove_duplicates_prefers_precedence():
    df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "y": [0, 0, 20],
            "x": [0, 3, 20],
            "score": [5, 10, 0],
        }
    )
    filtered = remove_duplicates(df.copy(), merge_radius=5.0, precedence="score", iterative=False)
    assert len(filtered) == 2
    assert set(filtered["score"]) == {10, 0}


def test_remove_duplicates_iterative_chain():
    # Points at x = 0, 1, 3 with merge_radius = 2.5 and scores 1, 2, 3:
    # Nearest neighbor of x=0 is x=1 (dist 1.0, score 1 < 2 -> x=0 removed).
    # Nearest neighbor of x=1 is x=0 (dist 1.0, score 2 > 1 -> x=1 kept in iter 1).
    # Nearest neighbor of x=3 is x=1 (dist 2.0, score 3 > 2 -> x=3 kept in iter 1).
    # In iter 2, remaining points are x=1 (score 2) and x=3 (score 3) at dist 2.0 <= 2.5:
    # Nearest neighbor of x=1 is now x=3 (score 2 < 3 -> x=1 removed in iter 2), leaving only x=3.
    df = pd.DataFrame(
        {
            "frame": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "x": [0.0, 1.0, 3.0],
            "score": [1.0, 2.0, 3.0],
        }
    )
    non_iter = remove_duplicates(df.copy(), merge_radius=2.5, precedence="score", iterative=False)
    assert len(non_iter) == 2

    iter_res = remove_duplicates(df.copy(), merge_radius=2.5, precedence="score", iterative=True)
    assert len(iter_res) == 1
    assert np.isclose(iter_res["score"].iloc[0], 3.0)


def test_add_nearest_neighbour_empty():
    empty = pd.DataFrame({"frame": [], "y": [], "x": []})
    result = add_nearest_neighbour(empty.copy())
    assert result.empty
    assert "nn_dist" in result.columns


def test_add_local_density_empty():
    empty = pd.DataFrame({"frame": [], "y": [], "x": []})
    result = add_local_density(empty.copy())
    assert result.empty
    assert "local_density" in result.columns


def test_remove_duplicates_empty():
    empty = pd.DataFrame({"frame": [], "y": [], "x": []})
    result = remove_duplicates(empty.copy(), merge_radius=5.0)
    assert result.empty
    assert all(col in result.columns for col in ["frame", "y", "x"])
