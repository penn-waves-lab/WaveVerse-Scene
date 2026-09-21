"""HMG path normalization and placement of motion in the scene."""

import numpy as np


def scene_transform(translation_xz, rotation_y):
    """Invert path normalization in Y-up coordinates, before the Y/Z swap."""
    angle = -rotation_y
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    transform[:3, 3] = [translation_xz[0], 0, translation_xz[1]]
    return transform


def mesh_scene_transform(translation_xz, rotation_y, coordinate_mapping="legacy_yz_swap"):
    """Convert local Y-up motion to world Z-up with the recorded convention.

    New motion uses a proper rotation, preserving anatomical left/right. Its
    conditioning path has the matching lateral sign from normalize_scene_path.
    Unmarked historical runs retain their original placement when read/repaired.
    """
    transform = scene_transform(translation_xz, rotation_y)[[0, 2, 1, 3], :]
    if coordinate_mapping == "right_handed":
        transform[:, 0] *= -1
    elif coordinate_mapping != "legacy_yz_swap":
        raise ValueError("Unknown motion coordinate mapping: " + coordinate_mapping)
    return transform


def normalize_scene_path(path):
    """Keep the world route while expressing it in the body's right-handed frame."""
    translation, rotation, normalized = normalize_sample_path(path)
    normalized[:, 0] *= -1
    return translation, rotation, normalized


def resample_trajectory(points, number=64):
    """
    Resample a 2D trajectory of shape (N, 2) to exactly 64 points
    with a constant arc-length spacing.

    Args:
        points: (N, 2) NumPy array of xy-coordinates.

    Returns:
        sampled: (number, 2) NumPy array of the resampled points.
    """
    # 1) Segment vectors & lengths
    seg_vectors = points[1:] - points[:-1]  # shape (N-1, 2)
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)  # shape (N-1,)

    # 2) Cumulative lengths (prefix sums), total arc length
    cum_lengths = np.cumsum(seg_lengths)  # shape (N-1,)
    total_length = cum_lengths[-1] if len(cum_lengths) > 0 else 0.0

    # Special case: If there's only 1 or 0 points, just pad or return as best we can
    if len(points) <= 1 or total_length == 0.0:
        return np.tile(points[0], (number, 1)) if len(points) > 0 else np.zeros((number, 2))

    # 3) Create number equally spaced arc-lengths from 0..total
    sample_distances = np.linspace(0, total_length, number)  # shape (64,)

    # 4) For each sample distance, find which segment it belongs to
    #    np.searchsorted returns indices such that cum_lengths[idx-1] < distance <= cum_lengths[idx].
    #    We'll clamp to ensure idx never goes out of range
    seg_indices = np.searchsorted(cum_lengths, sample_distances, side="right")
    seg_indices = np.clip(seg_indices, 0, len(seg_vectors) - 1)  # shape (64,)

    # 5) Distance at the start of each chosen segment.  For segment i, that start is cum_lengths[i-1],
    #    except for i=0 (start=0).
    seg_starts = np.zeros_like(seg_indices, dtype=float)
    seg_starts[seg_indices > 0] = cum_lengths[seg_indices[seg_indices > 0] - 1]

    # 6) Fraction along each segment
    seg_dist = sample_distances - seg_starts
    seg_len_for_sample = seg_lengths[seg_indices] + 1e-15  # avoid div-by-zero
    frac = seg_dist / seg_len_for_sample  # shape (64,)

    # 7) Interpolate between the segment's endpoints:
    #    left_pt = points[seg_idx], right_pt = points[seg_idx + 1]
    left_pts = points[seg_indices]  # shape (64, 2)
    right_pts = points[seg_indices + 1]  # shape (64, 2)

    sampled = left_pts + frac[:, None] * (right_pts - left_pts)  # shape (64, 2)

    return sampled


def normalize_sample_path(path, sample=64):
    """
    Normalize a path to have a fixed resolution.
    """

    translation_comp = path[0:1]
    path = path - translation_comp

    rotation_comp = np.arctan2(path[1, 1], path[1, 0])
    rotation_comp -= np.pi / 2

    rotation_matrix = np.array(
        [
            [np.cos(rotation_comp), -np.sin(rotation_comp)],
            [np.sin(rotation_comp), np.cos(rotation_comp)],
        ]
    )
    path = np.dot(path, rotation_matrix)

    path = resample_trajectory(path, number=64)

    return translation_comp, rotation_comp, path
